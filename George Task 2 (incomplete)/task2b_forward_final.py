#!/usr/bin/env python3
"""
AME 556 Final Project - Task 2b: Forward Walking
Requirements: Walk forward ≥0.5 m/s for ≥5 seconds
"""

import os, sys
if sys.platform == 'linux':
    os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import mujoco as mj
import imageio
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("/home/claude")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

L = 0.22
FOOT_R = 0.02
HIP_OFFSET = 0.125
HIP_TAU_MAX, KNEE_TAU_MAX = 30.0, 60.0
HIP_ANGLE_MIN, HIP_ANGLE_MAX = np.deg2rad(-120), np.deg2rad(30)
KNEE_ANGLE_MIN, KNEE_ANGLE_MAX = np.deg2rad(0), np.deg2rad(160)
HIP_VEL_MAX, KNEE_VEL_MAX = 30.0, 15.0
DT, FPS = 0.0005, 60

XML_MODEL = """
<mujoco model="biped2d_ame556">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.0005">
    <flag override="enable"/>
  </option>
  <default>
    <joint damping="0.5"/>
    <geom contype="1" conaffinity="1" condim="3" 
          solref="0.00001 1" solimp="0.95 0.99 0.0001" friction="0.5 0.5 0.1"/>
  </default>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.1 0.1 0.2" width="512" height="3072"/>
    <texture type="2d" name="gt" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" width="300" height="300"/>
    <material name="gm" texture="gt" texuniform="true" texrepeat="5 5"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <camera name="side" pos="0 -2.5 0.8" xyaxes="1 0 0 0 0 1" mode="trackcom"/>
    <geom name="floor" type="plane" size="100 5 0.1" material="gm"/>
    <body name="trunk" pos="0 0 0.5">
      <joint name="rx" type="slide" axis="1 0 0" damping="0"/>
      <joint name="rz" type="slide" axis="0 0 1" damping="0"/>
      <joint name="rt" type="hinge" axis="0 1 0" damping="0"/>
      <geom type="box" size="0.075 0.04 0.125" mass="8.0" rgba="0.2 0.6 0.9 1" 
            contype="0" conaffinity="0"/>
      <body name="left_thigh" pos="0 0.001 -0.125">
        <joint name="q1" type="hinge" axis="0 1 0" range="-2.094 0.524"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.025" mass="0.25" 
              rgba="0.3 0.8 0.3 1" contype="0" conaffinity="0"/>
        <inertial pos="0 0 -0.11" mass="0.25" diaginertia="0.001 0.001 0.0001"/>
        <body name="left_shank" pos="0 0 -0.22">
          <joint name="q2" type="hinge" axis="0 1 0" range="0 2.793"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="0.25" 
                rgba="0.3 0.8 0.3 1" contype="0" conaffinity="0"/>
          <inertial pos="0 0 -0.11" mass="0.25" diaginertia="0.001 0.001 0.0001"/>
          <geom name="left_foot" type="sphere" pos="0 0 -0.22" size="0.02" 
                rgba="0.9 0.6 0.2 1" mass="0.01"/>
        </body>
      </body>
      <body name="right_thigh" pos="0 -0.001 -0.125">
        <joint name="q3" type="hinge" axis="0 1 0" range="-2.094 0.524"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.025" mass="0.25" 
              rgba="0.8 0.3 0.3 1" contype="0" conaffinity="0"/>
        <inertial pos="0 0 -0.11" mass="0.25" diaginertia="0.001 0.001 0.0001"/>
        <body name="right_shank" pos="0 0 -0.22">
          <joint name="q4" type="hinge" axis="0 1 0" range="0 2.793"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="0.25" 
                rgba="0.8 0.3 0.3 1" contype="0" conaffinity="0"/>
          <inertial pos="0 0 -0.11" mass="0.25" diaginertia="0.001 0.001 0.0001"/>
          <geom name="right_foot" type="sphere" pos="0 0 -0.22" size="0.02" 
                rgba="0.9 0.6 0.2 1" mass="0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="q1" gear="1" ctrllimited="true" ctrlrange="-30 30"/>
    <motor joint="q2" gear="1" ctrllimited="true" ctrlrange="-60 60"/>
    <motor joint="q3" gear="1" ctrllimited="true" ctrlrange="-30 30"/>
    <motor joint="q4" gear="1" ctrllimited="true" ctrlrange="-60 60"/>
  </actuator>
</mujoco>
"""

def clamp_velocities(data, dof_indices):
    for i, dof in enumerate(dof_indices):
        if i in [0, 2]:
            data.qvel[dof] = np.clip(data.qvel[dof], -HIP_VEL_MAX, HIP_VEL_MAX)
        else:
            data.qvel[dof] = np.clip(data.qvel[dof], -KNEE_VEL_MAX, KNEE_VEL_MAX)

def saturate_torques(tau):
    tau_sat = tau.copy()
    tau_sat[0] = np.clip(tau_sat[0], -HIP_TAU_MAX, HIP_TAU_MAX)
    tau_sat[1] = np.clip(tau_sat[1], -KNEE_TAU_MAX, KNEE_TAU_MAX)
    tau_sat[2] = np.clip(tau_sat[2], -HIP_TAU_MAX, HIP_TAU_MAX)
    tau_sat[3] = np.clip(tau_sat[3], -KNEE_TAU_MAX, KNEE_TAU_MAX)
    return tau_sat

def get_foot_positions(model, data):
    lf_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "left_foot")
    rf_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "right_foot")
    return data.geom_xpos[lf_id].copy(), data.geom_xpos[rf_id].copy()

def run_forward_walking():
    print("=" * 70)
    print("Task 2b: Forward Walking")
    print("Requirement: ≥0.5 m/s for ≥5 seconds")
    print("=" * 70)
    
    model = mj.MjModel.from_xml_string(XML_MODEL)
    data = mj.MjData(model)
    
    # Stable asymmetric stance
    q1_init = np.deg2rad(-55)
    q2_init = np.deg2rad(50)
    q3_init = np.deg2rad(-2)
    q4_init = np.deg2rad(60)
    
    lf_z = -L * np.cos(q1_init) - L * np.cos(q1_init + q2_init)
    rf_z = -L * np.cos(q3_init) - L * np.cos(q3_init + q4_init)
    trunk_z = HIP_OFFSET - min(lf_z, rf_z) + FOOT_R + 0.01
    
    data.qpos[0] = 0
    data.qpos[1] = trunk_z - 0.5
    data.qpos[2] = 0
    data.qpos[3:7] = [q1_init, q2_init, q3_init, q4_init]
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    
    for _ in range(200):
        data.ctrl[:] = 0
        mj.mj_step(model, data)
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    
    q_base = data.qpos[3:7].copy()
    x_start = data.qpos[0]
    data.time = 0
    
    renderer = mj.Renderer(model, width=1280, height=720)
    cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "side")
    writer = imageio.get_writer(str(OUTPUT_DIR / "task2b_forward.mp4"), fps=FPS)
    steps_per_frame = int(1.0 / (FPS * DT))
    
    GAIT_FREQ = 1.4
    HIP_SWING_BASE = np.deg2rad(16)
    HIP_SWING_MAX = np.deg2rad(28)
    KNEE_LIFT = np.deg2rad(22)
    
    kp = np.array([380, 300, 380, 300])
    kd = np.array([38, 30, 38, 30])
    KP_PITCH = 160
    KD_PITCH = 28
    
    log_t, log_x, log_z, log_pitch = [], [], [], []
    log_q, log_tau, log_speed = [], [], []
    log_lf_x, log_rf_x = [], []
    
    step_count = 0
    duration = 12.0
    
    print(f"Initial stance: q1={np.rad2deg(q1_init):.0f}°, q2={np.rad2deg(q2_init):.0f}°, "
          f"q3={np.rad2deg(q3_init):.0f}°, q4={np.rad2deg(q4_init):.0f}°")
    
    while data.time < duration:
        t = data.time
        q = data.qpos[3:7]
        dq = data.qvel[3:7]
        x = data.qpos[0]
        z = data.qpos[1] + 0.5
        pitch = data.qpos[2]
        dpitch = data.qvel[2]
        vx = data.qvel[0]
        
        if abs(pitch) > np.deg2rad(55) or z < 0.15:
            print(f"FELL at t={t:.2f}s: pitch={np.rad2deg(pitch):.0f}°, z={z:.2f}m")
            break
        
        lf_pos, rf_pos = get_foot_positions(model, data)
        
        ramp = min(1.0, t / 2.5)
        swing_ramp = min(1.0, t / 6.0)
        hip_swing = HIP_SWING_BASE + (HIP_SWING_MAX - HIP_SWING_BASE) * swing_ramp
        
        phase = 2 * np.pi * GAIT_FREQ * t
        hip_osc = hip_swing * np.sin(phase) * ramp
        
        norm_phase = (phase % (2 * np.pi)) / np.pi
        if norm_phase < 1:
            left_lift = np.sin(np.pi * norm_phase)
            right_lift = 0
        else:
            right_lift = np.sin(np.pi * (norm_phase - 1))
            left_lift = 0
        
        left_knee_offset = KNEE_LIFT * left_lift * ramp
        right_knee_offset = KNEE_LIFT * right_lift * ramp
        
        q_des = np.array([
            q_base[0] - hip_osc,
            q_base[1] + left_knee_offset,
            q_base[2] + hip_osc,
            q_base[3] + right_knee_offset
        ])
        
        tau = kp * (q_des - q) - kd * dq
        
        base_lean = np.deg2rad(5)
        velocity_correction = -0.03 * vx
        lean = (base_lean + velocity_correction) * ramp
        lean = np.clip(lean, np.deg2rad(2), np.deg2rad(12))
        
        tau_pitch = -KP_PITCH * (pitch - lean) - KD_PITCH * dpitch
        tau[0] += tau_pitch * 0.5
        tau[2] += tau_pitch * 0.5
        
        tau[0] += 10.5 * ramp
        tau[2] += 10.5 * ramp
        
        data.qacc[:] = 0
        mj.mj_inverse(model, data)
        tau += data.qfrc_inverse[3:7]
        
        clamp_velocities(data, [3, 4, 5, 6])
        tau = saturate_torques(tau)
        
        data.ctrl[:] = tau
        mj.mj_step(model, data)
        
        log_t.append(t)
        log_x.append(x)
        log_z.append(z)
        log_pitch.append(pitch)
        log_q.append(q.copy())
        log_tau.append(tau.copy())
        log_speed.append(vx)
        log_lf_x.append(lf_pos[0])
        log_rf_x.append(rf_pos[0])
        
        if step_count % steps_per_frame == 0:
            renderer.update_scene(data, camera=cam_id)
            writer.append_data(renderer.render())
        step_count += 1
    
    writer.close()
    renderer.close()
    
    final_time = data.time
    distance = data.qpos[0] - x_start
    avg_speed = distance / max(final_time, 0.01)
    
    lf_x_arr = np.array(log_lf_x)
    rf_x_arr = np.array(log_rf_x)
    crossings = np.sum(np.diff(np.sign(lf_x_arr - rf_x_arr)) != 0)
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"  Duration: {final_time:.2f}s (requirement: ≥5s)")
    print(f"  Distance: {distance:.3f}m")
    print(f"  Average speed: {avg_speed:.3f} m/s (requirement: ≥0.5 m/s)")
    print(f"  Foot crossings: {crossings}")
    
    success = final_time >= 5.0 and avg_speed >= 0.5
    print(f"\n  *** {'PASS' if success else 'FAIL'} ***")
    print(f"{'='*60}")
    
    # Create plot
    log_q = np.array(log_q)
    log_tau = np.array(log_tau)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'Task 2b: Forward Walking - {distance:.2f}m in {final_time:.1f}s = {avg_speed:.3f} m/s', fontsize=14)
    
    axes[0,0].plot(log_t, log_x, 'b-', lw=2)
    axes[0,0].set_ylabel('COM X (m)')
    axes[0,0].set_title('Position')
    axes[0,0].grid(True)
    
    axes[0,1].plot(log_t, log_speed, 'g-', lw=1)
    axes[0,1].axhline(0.5, color='r', ls='--', label='Target: 0.5 m/s')
    axes[0,1].set_ylabel('Speed (m/s)')
    axes[0,1].set_title('Forward Speed')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    axes[1,0].plot(log_t, np.rad2deg(log_q[:,0]), 'g-', label='q1 (L hip)', lw=1.5)
    axes[1,0].plot(log_t, np.rad2deg(log_q[:,2]), 'r-', label='q3 (R hip)', lw=1.5)
    axes[1,0].axhline(-120, color='k', ls=':', alpha=0.5)
    axes[1,0].axhline(30, color='k', ls=':', alpha=0.5)
    axes[1,0].set_ylabel('Angle (°)')
    axes[1,0].set_title('Hip Angles (limits: -120° to 30°)')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    axes[1,1].plot(log_t, np.rad2deg(log_q[:,1]), 'g-', label='q2 (L knee)', lw=1.5)
    axes[1,1].plot(log_t, np.rad2deg(log_q[:,3]), 'r-', label='q4 (R knee)', lw=1.5)
    axes[1,1].axhline(0, color='k', ls=':', alpha=0.5)
    axes[1,1].axhline(160, color='k', ls=':', alpha=0.5)
    axes[1,1].set_ylabel('Angle (°)')
    axes[1,1].set_title('Knee Angles (limits: 0° to 160°)')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    axes[2,0].plot(log_t, log_tau[:,0], 'g-', label='τ1', lw=1)
    axes[2,0].plot(log_t, log_tau[:,2], 'r-', label='τ3', lw=1)
    axes[2,0].axhline(-30, color='k', ls=':', alpha=0.5)
    axes[2,0].axhline(30, color='k', ls=':', alpha=0.5)
    axes[2,0].set_ylabel('Torque (Nm)')
    axes[2,0].set_xlabel('Time (s)')
    axes[2,0].set_title('Hip Torques (limit: ±30 Nm)')
    axes[2,0].legend()
    axes[2,0].grid(True)
    
    axes[2,1].plot(log_t, log_tau[:,1], 'g-', label='τ2', lw=1)
    axes[2,1].plot(log_t, log_tau[:,3], 'r-', label='τ4', lw=1)
    axes[2,1].axhline(-60, color='k', ls=':', alpha=0.5)
    axes[2,1].axhline(60, color='k', ls=':', alpha=0.5)
    axes[2,1].set_ylabel('Torque (Nm)')
    axes[2,1].set_xlabel('Time (s)')
    axes[2,1].set_title('Knee Torques (limit: ±60 Nm)')
    axes[2,1].legend()
    axes[2,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "task2b_forward_plot.png"), dpi=150)
    plt.close()
    
    print(f"\nVideo: {OUTPUT_DIR}/task2b_forward.mp4")
    print(f"Plot: {OUTPUT_DIR}/task2b_forward_plot.png")
    
    return {'duration': final_time, 'distance': distance, 'avg_speed': avg_speed, 'success': success}

if __name__ == "__main__":
    run_forward_walking()
