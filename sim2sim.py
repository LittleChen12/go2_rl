# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
import time
import math
import numpy as np
import mujoco
import mujoco.viewer
import glfw
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from global_config import ROOT_DIR
from configs.tinymal_constraint_him import TinymalConstraintHimRoughCfg
from configs.go2_constraint_him import Go2ConstraintHimRoughCfg
import torch
from pynput import keyboard

import lcm
from go2_lcm.lcm_type.go2_lcm import Request, Response
lcm = lcm.LCM()
import threading, queue

# 0. 选定 device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# default_dof_pos=[-0.16,0.68,1.3 ,0.16,0.68,1.3, -0.16,0.68,1.3, 0.16,0.68,1.3]#默认角度需要与isacc一致
default_dof_pos = [
    0.1,   # FL_hip_joint
    0.8,   # FL_thigh_joint
   -1.5,   # FL_calf_joint
   -0.1,   # FR_hip_joint
    0.8,   # FR_thigh_joint
   -1.5,   # FR_calf_joint
    0.1,   # RL _hip_joint
    1.0,   # RL_thigh_joint
   -1.5,   # RL_calf_joint
   -0.1,   # RR_hip_joint
    1.0,   # RR_thigh_joint
   -1.5    # RR_calf_joint
]

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

class KeyController:
    def __init__(self):
        self.vx   = 0.0   # 前后
        self.vy   = 0.0   # 左右
        self.dyaw = 0.0   # 偏航

    def on_press(self, key):
        # 方向键按下 —— 前后左右
        if key == keyboard.Key.up:
            self.vx = +1.0
        elif key == keyboard.Key.down:
            self.vx = -1.0
        elif key == keyboard.Key.left:
            self.vy = +1.0
        elif key == keyboard.Key.right:
            self.vy = -1.0

        # Q/E 按下 —— 偏航
        try:
            c = key.char.lower()
        except AttributeError:
            c = None

        if c == 'q':
            self.dyaw = +1.0
        elif c == 'e':
            self.dyaw = -1.0

    def on_release(self, key):
        # 方向键松开 —— 前后或左右归零
        if key in (keyboard.Key.up, keyboard.Key.down):
            self.vx = 0.0
        elif key in (keyboard.Key.left, keyboard.Key.right):
            self.vy = 0.0

        # Q/E 松开 —— 偏航归零
        try:
            c = key.char.lower()
        except AttributeError:
            c = None

        if c in ('q', 'e'):
            self.dyaw = 0.0

    def start(self):
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        listener.daemon = True
        listener.start()

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    """
    Extracts an observation from the mujoco data structure
    """
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def _low_pass_action_filter(actions,last_actions):
    actons_filtered = last_actions * 0.2 + actions * 0.8
    return actons_filtered

def run_mujoco(policy, cfg):
    global default_dof_pos

    # ———————————————— 1) 线程同步 & 共享数据 ————————————————
    data_lock = threading.Lock()
    stop_event = threading.Event()

    shared = {
        'omega': None,
        'eu_ang': None,
        'cmd': None,
        'q': None,
        'dq': None,
        'hist_obs': None,
        'last_actions': np.zeros(cfg.env.num_actions, dtype=np.double),
        'action_flt': np.zeros(cfg.env.num_actions, dtype=np.double),  # 新增：最新滤波后的动作
        'target_q': np.zeros(cfg.env.num_actions, dtype=np.double),
    }

    # 启用 policy 到 eval 模式
    device = next(policy.parameters()).device
    policy = policy.to(device).eval()

    # ———————————————— 2) 推理 + LCM 发布线程 ————————————————
    def infer_and_publish():
        period = 1.0 / 50.0  # 50 Hz

        while not stop_event.is_set():
            t0 = time.time()
            with data_lock:
                if shared['omega'] is None:
                    to_proc = False
                else:
                    # 拷贝一份，断开与主线程共享
                    omega = shared['omega'].copy()
                    eu_ang = shared['eu_ang'].copy()
                    cmd = shared['cmd']
                    q = shared['q'].copy()
                    dq = shared['dq'].copy()
                    hist_list = list(shared['hist_obs'])
                    last_act = shared['last_actions'].copy()
                    action_flt = shared['action_flt'].copy()  # 最新滤波后的动作
                    target_q_prev = shared['target_q'].copy()
                    to_proc = True

            if to_proc:
                # 1) 构建当前 obs 向量
                obs = np.zeros([1, cfg.env.n_proprio], dtype=np.float32)
                obs[0, 0:3] = omega * cfg.normalization.obs_scales.ang_vel
                obs[0, 3:6] = eu_ang * cfg.normalization.obs_scales.quat
                obs[0, 6] = cmd.vx * cfg.normalization.obs_scales.lin_vel
                obs[0, 7] = cmd.vy * cfg.normalization.obs_scales.lin_vel
                obs[0, 8] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
                obs[0, 9:21] = (q - default_dof_pos) * cfg.normalization.obs_scales.dof_pos
                obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 33:45] = action_flt
                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

                # 2) 构建 history buffer
                n_pr = cfg.env.n_proprio
                n_pl = cfg.env.n_priv_latent
                n_sc = cfg.env.n_scan
                hlen = cfg.env.history_len
                num_obs = cfg.env.num_observations

                policy_input = np.zeros([1, num_obs], dtype=np.float16)
                hist_buf = np.zeros([1, hlen * n_pr], dtype=np.float16)

                # 填入当前 obs
                policy_input[0, 0:n_pr] = obs

                # 填入 history
                for i in range(hlen):
                    start = n_pr + n_pl + n_sc + i * n_pr
                    policy_input[0, start:start + n_pr] = hist_list[i][0, :]
                    hist_buf[0, i * n_pr:(i + 1) * n_pr] = hist_list[i][0, :]

                # 3) 调用 policy 推理
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                hist_tensor = torch.tensor(hist_buf, dtype=torch.float32, device=device).view(1, hlen, n_pr)
                with torch.no_grad():
                    out = policy(obs_tensor, hist_tensor)[0].cpu().numpy()

                action = out
                action_flt = _low_pass_action_filter(action, last_act)
                new_target = action_flt * cfg.control.action_scale + default_dof_pos

                # 4) 更新共享 target 和 last_actions
                with data_lock:
                    shared['last_actions'] = last_act
                    shared['target_q'] = new_target
                    shared['action_flt'] = action_flt

                # 5) 发布 LCM 消息
                msg = Request()
                msg.omega = omega
                msg.eu_ang = eu_ang
                msg.command[0] = cmd.vx
                msg.command[1] = cmd.vy
                msg.command[2] = cmd.dyaw
                msg.q = q
                msg.dq = dq
                lcm.publish("LCM_OBS", msg.encode())

            # 精确睡眠
            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    thread = threading.Thread(target=infer_and_publish, daemon=True)
    thread.start()

    # ———————————————— 3) 原仿真初始化 ————————————————
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = cfg.sim_config.dt  # 1 kHz

    target_q = shared['target_q']  # 初始零
    last_act = shared['last_actions']

    hist_obs = deque(
        [np.zeros([1, cfg.env.n_proprio], dtype=np.float32)
         for _ in range(cfg.env.history_len)],
        maxlen=cfg.env.history_len
    )

    key_ctrl = KeyController()
    key_ctrl.start()
    counter = 0
    # ———————————————— 4) 主循环（1 kHz） ————————————————
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t_start = time.time()
            # A) PD 控制 & 物理步进
            tau = pd_control(
                target_q,
                data.qpos[7:], cfg.robot_config.kps,
                np.zeros_like(cfg.robot_config.kds),
                data.qvel[6:], cfg.robot_config.kds
            )
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            counter += 1

            # B) 每隔 decimation 更新共享观测
            if counter % cfg.sim_config.decimation == 0:
                with data_lock:
                    action_flt = shared['action_flt'].copy()
                omega = data.qvel[3:6].copy()
                q = data.qpos[7:].copy()
                dq = data.qvel[6:].copy()
                quat = data.qpos[3:7].copy()

                eu_ang = quaternion_to_euler_array(quat)
                eu_ang[eu_ang > np.pi] -= 2 * np.pi
                # 读取键盘命令
                cmd.vx   = key_ctrl.vx
                cmd.vy   = key_ctrl.vy
                cmd.dyaw = key_ctrl.dyaw

                # 更新 history buffer（只为线程准备）
                obs = np.zeros([1, cfg.env.n_proprio], dtype=np.float32)
                obs[0, 0:3] = omega * cfg.normalization.obs_scales.ang_vel
                obs[0, 3:6] = eu_ang * cfg.normalization.obs_scales.quat
                obs[0, 6] = cmd.vx * cfg.normalization.obs_scales.lin_vel
                obs[0, 7] = cmd.vy * cfg.normalization.obs_scales.lin_vel
                obs[0, 8] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
                obs[0, 9:21] = (q - default_dof_pos) * cfg.normalization.obs_scales.dof_pos
                obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 33:45] = action_flt
                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
                hist_obs.append(obs)

                # 写入共享数据：omega, eu_ang, cmd, q, dq, hist_obs
                with data_lock:
                    shared['omega'] = omega
                    shared['eu_ang'] = eu_ang
                    shared['cmd'] = cmd
                    shared['q'] = q
                    shared['dq'] = dq
                    shared['hist_obs'] = list(hist_obs)
                    # 主线程拿最新 target_q 用于 PD 控制
                    target_q = shared['target_q']
                    last_act = shared['last_actions']

            # C) 渲染 10hz
            if counter % 10 == 0:
                viewer.sync()
            to_sleep = model.opt.timestep - (time.time() - t_start)
            if to_sleep > 0:
                time.sleep(to_sleep)

    # ———————————————— 5) 退出前清理线程 ————————————————
    stop_event.set()
    thread.join()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default='/home/zhu/LocomotionWithNP3O-master/model_jitt.pt',
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', default=False)
    args = parser.parse_args()

    class Sim2simCfg(Go2ConstraintHimRoughCfg):    # Go2ConstraintHimRoughCfg

        class sim_config:
            if args.terrain:
                # mujoco_model_path = f'{ROOT_DIR}/resources/tinymal/xml/world_terrain.xml'
                mujoco_model_path = f'{ROOT_DIR}/resources/go2/mujoco/scene.xml'
            else:
                # mujoco_model_path = f'{ROOT_DIR}/resources/tinymal/xml/world.xml'
                mujoco_model_path = f'{ROOT_DIR}/resources/go2/mujoco/scene.xml'
            sim_duration = 60.0
            dt = 0.001 #1Khz底层
            decimation = 20 # 50Hz

        class robot_config:
            # kp_all = 3.5
            # kd_all = 0.15
            kp_all = 30
            kd_all = 0.75
            kps = np.array([kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all], dtype=np.double)#PD和isacc内部一致
            kds = np.array([kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all], dtype=np.double)
            tau_limit = 15. * np.ones(12, dtype=np.double)#nm

    # policy = torch.load(args.load_model)
    # 1. 加载到 GPU 并转 float32
    policy = torch.jit.load(args.load_model) #jit模型
    policy = policy.to(device).float().eval()
    print("Model dtype:", next(policy.parameters()).dtype)
    run_mujoco(policy, Sim2simCfg())
