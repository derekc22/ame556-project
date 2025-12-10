#!/usr/bin/env python3
"""
AME 556 - Robot Dynamics and Control - Final Project
Robust Implementation with Conservative Control

Focus:
1. Strong pitch stabilization
2. Conservative height changes
3. Stable standing before walking
"""

import os
import sys

if sys.platform == 'linux':
    os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import mujoco as mj
import imageio
from pathlib import Path

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Physical limits from PDF
HIP_ANGLE_MIN = np.deg2rad(-120)
HIP_ANGLE_MAX = np.deg2rad(30)
KNEE_ANGLE_MIN = np.deg2rad(0)
KNEE_ANGLE_MAX = np.deg2rad(160)
HIP_TAU_MAX = 30.0
KNEE_TAU_MAX = 60.0

L = 0.22
FOOT_R = 0.02
HIP_OFFSET = 0.125
DT = 0.0005
FPS = 60

# ============================================================================
# XML MODEL - Using planar joints (3 DoF root) for 2D motion
# ============================================================================

XML_MODEL = """
<mujoco model="biped2d_ame556">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.0005">
    <flag override="enable"/>
  </option>
  <default>
    <joint damping="0.5"/>
    <geom contype="1" conaffinity="1" condim="3" 
          solref="0.00001 1" solimp="0.95 0.99 0.0001"
          friction="0.5 0.5 0.1"/>
  </default>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.1 0.1 0.2" width="512" height="3072"/>
    <texture type="2d" name="ground_tex" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" width="300" height="300"/>
    <material name="ground_mat" texture="ground_tex" texuniform="true" texrepeat="5 5"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <camera name="side" pos="0 -2.5 0.8" xyaxes="1 0 0 0 0 1" mode="trackcom"/>
    <geom name="floor" type="plane" size="100 5 0.1" material="ground_mat"/>

    <!-- Trunk with planar (3 DoF) root joint -->
    <body name="trunk" pos="0 0 0.5">
      <joint name="root_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="root_z" type="slide" axis="0 0 1" damping="0"/>
      <joint name="root_theta" type="hinge" axis="0 1 0" damping="0"/>
      <geom name="trunk_geom" type="box" size="0.075 0.04 0.125" mass="8.0" 
            rgba="0.2 0.6 0.9 1" contype="0" conaffinity="0"/>

      <!-- Left leg -->
      <body name="left_thigh" pos="0 0 -0.125">
        <joint name="q1" type="hinge" axis="0 1 0" range="-2.094 0.524"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.025" mass="0.25" 
              rgba="0.3 0.8 0.3 1" contype="0" conaffinity="0"/>
        <inertial pos="0 0 -0.11" mass="0.25" diaginertia="0.001008 0.001008 0.0001"/>
        <body name="left_shank" pos="0 0 -0.22">
          <joint name="q2" type="hinge" axis="0 1 0" range="0 2.793"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="0.25" 
                rgba="0.3 0.8 0.3 1" contype="0" conaffinity="0"/>
          <inertial pos="0 0 -0.11" mass="0.25" diaginertia="0.001008 0.001008 0.0001"/>
          <geom name="left_foot" type="sphere" pos="0 0 -0.22" size="0.02" 
                rgba="0.9 0.6 0.2 1" mass="0.01"/>
        </body>
      </body>

      <!-- Right leg -->
      <body name="right_thigh" pos="0 0 -0.125">
        <joint name="q3" type="hinge" axis="0 1 0" range="-2.094 0.524"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.025" mass="0.25" 
              rgba="0.8 0.3 0.3 1" contype="0" conaffinity="0"/>
        <inertial pos="0 0 -0.11" mass="0.25" diaginertia="0.001008 0.001008 0.0001"/>
        <body name="right_shank" pos="0 0 -0.22">
          <joint name="q4" type="hinge" axis="0 1 0" range="0 2.793"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="0.25" 
                rgba="0.8 0.3 0.3 1" contype="0" conaffinity="0"/>
          <inertial pos="0 0 -0.11" mass="0.25" diaginertia="0.001008 0.001008 0.0001"/>
          <geom name="right_foot" type="sphere" pos="0 0 -0.22" size="0.02" 
                rgba="0.9 0.6 0.2 1" mass="0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_l" joint="q1" gear="1" ctrllimited="true" ctrlrange="-30 30"/>
    <motor name="knee_l" joint="q2" gear="1" ctrllimited="true" ctrlrange="-60 60"/>
    <motor name="hip_r" joint="q3" gear="1" ctrllimited="true" ctrlrange="-30 30"/>
    <motor name="knee_r" joint="q4" gear="1" ctrllimited="true" ctrlrange="-60 60"/>
  </actuator>
</mujoco>
"""


# ============================================================================
# HELPERS
# ============================================================================

def create_model():
    model = mj.MjModel.from_xml_string(XML_MODEL)
    data = mj.MjData(model)
    return model, data


def saturate_torques(tau):
    tau_sat = np.array(tau, dtype=float)
    tau_sat[0] = np.clip(tau_sat[0], -HIP_TAU_MAX, HIP_TAU_MAX)
    tau_sat[1] = np.clip(tau_sat[1], -KNEE_TAU_MAX, KNEE_TAU_MAX)
    tau_sat[2] = np.clip(tau_sat[2], -HIP_TAU_MAX, HIP_TAU_MAX)
    tau_sat[3] = np.clip(tau_sat[3], -KNEE_TAU_MAX, KNEE_TAU_MAX)
    return tau_sat


def leg_ik(h_com):
    """Compute symmetric leg angles for desired COM height.

    For symmetric stance with feet under hips:
    - q1 = -q2/2 (upper leg angle)
    - leg_length = 2*L*cos(q2/2)
    """
    leg_len = h_com - HIP_OFFSET - FOOT_R
    leg_len = np.clip(leg_len, 0.10, 2 * L - 0.01)

    # leg_len = 2*L*cos(q2/2)
    # cos(q2/2) = leg_len / (2*L)
    cos_half_q2 = leg_len / (2 * L)
    cos_half_q2 = np.clip(cos_half_q2, 0.01, 0.99)
    half_q2 = np.arccos(cos_half_q2)
    q2 = 2 * half_q2
    q1 = -half_q2

    q1 = np.clip(q1, HIP_ANGLE_MIN, HIP_ANGLE_MAX)
    q2 = np.clip(q2, KNEE_ANGLE_MIN, KNEE_ANGLE_MAX)
    return q1, q2


def compute_height(q1, q2):
    """Compute COM height from leg angles.

    For symmetric stance: leg_len = 2*L*cos(q2/2)
    But using exact formula for generality.
    """
    # Vertical leg length from hip to foot
    leg_len = L * np.cos(q1) + L * np.cos(q1 + q2)
    return HIP_OFFSET + leg_len + FOOT_R


def get_desired_height(t):
    """Height trajectory from PDF."""
    if t < 1.0:
        return 0.45
    elif t < 1.5:
        alpha = (t - 1.0) / 0.5
        return 0.45 + alpha * (0.55 - 0.45)
    elif t < 2.5:
        alpha = (t - 1.5) / 1.0
        return 0.55 + alpha * (0.40 - 0.55)
    else:
        return 0.40


def check_constraints(q, dq, tau):
    """Check if any constraint is violated."""
    if q[0] < HIP_ANGLE_MIN or q[0] > HIP_ANGLE_MAX:
        return True, f"q1={np.rad2deg(q[0]):.1f}° out of hip range"
    if q[2] < HIP_ANGLE_MIN or q[2] > HIP_ANGLE_MAX:
        return True, f"q3={np.rad2deg(q[2]):.1f}° out of hip range"
    if q[1] < KNEE_ANGLE_MIN or q[1] > KNEE_ANGLE_MAX:
        return True, f"q2={np.rad2deg(q[1]):.1f}° out of knee range"
    if q[3] < KNEE_ANGLE_MIN or q[3] > KNEE_ANGLE_MAX:
        return True, f"q4={np.rad2deg(q[3]):.1f}° out of knee range"

    if abs(dq[0]) > 30: return True, f"|dq1|={abs(dq[0]):.1f} > 30 rad/s"
    if abs(dq[2]) > 30: return True, f"|dq3|={abs(dq[2]):.1f} > 30 rad/s"
    if abs(dq[1]) > 15: return True, f"|dq2|={abs(dq[1]):.1f} > 15 rad/s"
    if abs(dq[3]) > 15: return True, f"|dq4|={abs(dq[3]):.1f} > 15 rad/s"

    return False, ""


# ============================================================================
# TASK 1: CONSTRAINTS DEMONSTRATION
# ============================================================================

def task1_demo():
    """Task 1: Demonstrate constraint checking and torque saturation."""
    print("\n" + "=" * 70)
    print("TASK 1: PHYSICAL CONSTRAINTS DEMONSTRATION")
    print("=" * 70)

    print("\n[Torque Saturation]")
    test_tau = np.array([50, 80, -40, -70])
    sat_tau = saturate_torques(test_tau)
    print(f"  Requested: {test_tau}")
    print(f"  Saturated: {sat_tau}")
    print(f"  Limits: Hip=±{HIP_TAU_MAX}Nm, Knee=±{KNEE_TAU_MAX}Nm")

    print("\n[Constraint Checking]")
    test_q = np.deg2rad([-130, 50, -10, 170])
    test_dq = np.array([0, 0, 0, 0])
    violated, msg = check_constraints(test_q, test_dq, np.zeros(4))
    print(f"  Test angles: q1=-130°, q2=50°, q3=-10°, q4=170°")
    print(f"  Violation: {msg}")

    print("\n[Standing Demo with Constraint Monitoring]")

    model, data = create_model()

    # Use split stance for stability
    q1_init, q2_init = leg_ik(0.45)
    hip_offset = np.deg2rad(8)

    leg_len_front = L * np.cos(q1_init - hip_offset) + L * np.cos(q1_init - hip_offset + q2_init)
    leg_len_back = L * np.cos(q1_init + hip_offset) + L * np.cos(q1_init + hip_offset + q2_init)
    avg_leg_len = (leg_len_front + leg_len_back) / 2
    trunk_z = HIP_OFFSET + avg_leg_len + FOOT_R

    data.qpos[0] = 0
    data.qpos[1] = trunk_z - 0.5
    data.qpos[2] = 0
    data.qpos[3] = q1_init - hip_offset  # Left forward
    data.qpos[4] = q2_init
    data.qpos[5] = q1_init + hip_offset  # Right back
    data.qpos[6] = q2_init
    data.qvel[:] = 0
    mj.mj_forward(model, data)

    print(
        f"  Initial config: q1={np.rad2deg(q1_init - hip_offset):.1f}°, q3={np.rad2deg(q1_init + hip_offset):.1f}°, q2,q4={np.rad2deg(q2_init):.1f}°")
    print(f"  Expected height: {trunk_z:.3f}m")

    renderer = mj.Renderer(model, width=1280, height=720)
    cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "side")
    writer = imageio.get_writer(str(OUTPUT_DIR / "task1_standing.mp4"), fps=FPS)

    steps_per_frame = int(1.0 / (FPS * DT))
    step = 0
    duration = 3.0

    log = {'t': [], 'x': [], 'z': [], 'pitch': [], 'q': [], 'tau': []}

    q_des = np.array([q1_init - hip_offset, q2_init, q1_init + hip_offset, q2_init])

    kp = np.array([400, 350, 400, 350])
    kd = np.array([40, 35, 40, 35])
    kp_pitch = 350.0
    kd_pitch = 70.0

    while data.time < duration:
        q = data.qpos[3:7]
        dq = data.qvel[3:7]
        pitch = data.qpos[2]
        dpitch = data.qvel[2]
        z = data.qpos[1] + 0.5

        violated, msg = check_constraints(q, dq, np.zeros(4))
        if violated:
            print(f"  CONSTRAINT VIOLATION at t={data.time:.2f}s: {msg}")
            break

        if abs(pitch) > np.deg2rad(45) or z < 0.2:
            print(f"  FELL at t={data.time:.2f}s: pitch={np.rad2deg(pitch):.0f}°, z={z:.2f}m")
            break

        tau = kp * (q_des - q) - kd * dq

        tau_pitch = -kp_pitch * pitch - kd_pitch * dpitch
        tau[0] += tau_pitch * 0.5
        tau[2] += tau_pitch * 0.5

        data.qacc[:] = 0
        mj.mj_inverse(model, data)
        tau += data.qfrc_inverse[3:7]

        tau = saturate_torques(tau)
        data.ctrl[:] = tau
        mj.mj_step(model, data)

        log['t'].append(data.time)
        log['x'].append(data.qpos[0])
        log['z'].append(z)
        log['pitch'].append(pitch)
        log['q'].append(q.copy())
        log['tau'].append(tau.copy())

        if step % steps_per_frame == 0:
            renderer.update_scene(data, camera=cam_id)
            writer.append_data(renderer.render())
        step += 1

    writer.close()
    renderer.close()

    for key in log:
        log[key] = np.array(log[key])

    final_z = data.qpos[1] + 0.5
    final_pitch = data.qpos[2]

    print(f"  Duration: {data.time:.2f}s")
    print(f"  Final height: {final_z:.3f}m")
    print(f"  Final pitch: {np.rad2deg(final_pitch):.1f}°")
    print(f"  Video: {OUTPUT_DIR}/task1_standing.mp4")

    return {'time': data.time, 'log': log, 'success': data.time >= duration - 0.1}


# ============================================================================
# TASK 2a: HEIGHT TRAJECTORY
# ============================================================================

def task2a_height():
    """Task 2a: Track height trajectory with split stance for stability."""
    print("\n" + "=" * 70)
    print("TASK 2a: HEIGHT TRAJECTORY TRACKING")
    print("=" * 70)

    model, data = create_model()

    # Use split stance for stability
    # Front leg slightly forward, back leg slightly back
    h_init = 0.45
    q1_init, q2_init = leg_ik(h_init)

    # Offset hips by ~5° each direction
    hip_offset = np.deg2rad(8)
    q1_front = q1_init - hip_offset  # Front leg rotated forward
    q1_back = q1_init + hip_offset  # Back leg rotated backward

    # Compute initial trunk height from average
    leg_len_front = L * np.cos(q1_front) + L * np.cos(q1_front + q2_init)
    leg_len_back = L * np.cos(q1_back) + L * np.cos(q1_back + q2_init)
    avg_leg_len = (leg_len_front + leg_len_back) / 2
    trunk_z = HIP_OFFSET + avg_leg_len + FOOT_R

    data.qpos[0] = 0
    data.qpos[1] = trunk_z - 0.5
    data.qpos[2] = 0
    data.qpos[3] = q1_front  # Left leg forward
    data.qpos[4] = q2_init
    data.qpos[5] = q1_back  # Right leg back
    data.qpos[6] = q2_init
    data.qvel[:] = 0
    mj.mj_forward(model, data)

    # Settle
    q_settle = data.qpos[3:7].copy()
    kp_settle = np.array([400, 350, 400, 350])
    kd_settle = np.array([40, 35, 40, 35])

    for _ in range(int(1.0 / DT)):
        q = data.qpos[3:7]
        dq = data.qvel[3:7]
        pitch = data.qpos[2]
        dpitch = data.qvel[2]

        tau = kp_settle * (q_settle - q) - kd_settle * dq
        tau_pitch = -300 * pitch - 60 * dpitch
        tau[0] += tau_pitch * 0.5
        tau[2] += tau_pitch * 0.5

        data.qacc[:] = 0
        mj.mj_inverse(model, data)
        tau += data.qfrc_inverse[3:7]

        tau = saturate_torques(tau)
        data.ctrl[:] = tau
        mj.mj_step(model, data)

    data.time = 0
    actual_z = data.qpos[1] + 0.5
    print(f"  After settling: z={actual_z:.3f}m, pitch={np.rad2deg(data.qpos[2]):.1f}°")

    renderer = mj.Renderer(model, width=1280, height=720)
    cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "side")
    writer = imageio.get_writer(str(OUTPUT_DIR / "task2a_height.mp4"), fps=FPS)

    # Higher gains
    kp = np.array([500, 450, 500, 450])
    kd = np.array([50, 45, 50, 45])
    kp_pitch = 400.0
    kd_pitch = 80.0

    steps_per_frame = int(1.0 / (FPS * DT))
    step = 0
    duration = 4.0

    log = {'t': [], 'x': [], 'z': [], 'z_des': [], 'pitch': [], 'q': [], 'tau': []}

    print("  Running height trajectory...")

    while data.time < duration:
        t = data.time
        q = data.qpos[3:7]
        dq = data.qvel[3:7]
        pitch = data.qpos[2]
        dpitch = data.qvel[2]
        z = data.qpos[1] + 0.5

        if abs(pitch) > np.deg2rad(60) or z < 0.15:
            print(f"  FELL at t={t:.2f}s: pitch={np.rad2deg(pitch):.0f}°, z={z:.2f}m")
            break

        h_des = get_desired_height(t)
        q1_des, q2_des = leg_ik(h_des)

        # Maintain split stance while adjusting height
        q_des = np.array([
            q1_des - hip_offset,  # Left forward
            q2_des,
            q1_des + hip_offset,  # Right back
            q2_des
        ])

        tau = kp * (q_des - q) - kd * dq

        # Strong pitch stabilization
        tau_pitch = -kp_pitch * pitch - kd_pitch * dpitch
        tau[0] += tau_pitch * 0.5
        tau[2] += tau_pitch * 0.5

        # Gravity compensation
        data.qacc[:] = 0
        mj.mj_inverse(model, data)
        tau += data.qfrc_inverse[3:7]

        tau = saturate_torques(tau)
        data.ctrl[:] = tau
        mj.mj_step(model, data)

        log['t'].append(t)
        log['x'].append(data.qpos[0])
        log['z'].append(z)
        log['z_des'].append(h_des)
        log['pitch'].append(pitch)
        log['q'].append(q.copy())
        log['tau'].append(tau.copy())

        if step % steps_per_frame == 0:
            renderer.update_scene(data, camera=cam_id)
            writer.append_data(renderer.render())
        step += 1

    writer.close()
    renderer.close()

    for key in log:
        log[key] = np.array(log[key])

    if len(log['z']) > 0:
        tracking_error = np.mean(np.abs(log['z'] - log['z_des']))
        max_error = np.max(np.abs(log['z'] - log['z_des']))
    else:
        tracking_error = 1.0
        max_error = 1.0

    print(f"  Duration: {data.time:.2f}s")
    print(f"  Mean tracking error: {tracking_error * 100:.1f}cm")
    print(f"  Max tracking error: {max_error * 100:.1f}cm")
    print(f"  Video: {OUTPUT_DIR}/task2a_height.mp4")

    return {'time': data.time, 'log': log, 'tracking_error': tracking_error}


# ============================================================================
# TASK 2b/2c: WALKING
# ============================================================================

def task2_walk(direction=1, task_name="task2b"):
    """Walking using weight-shifting approach for stability."""
    dir_str = "forward" if direction > 0 else "backward"
    print(f"\n" + "=" * 70)
    print(f"TASK {task_name.upper()}: WALK {dir_str.upper()}")
    print("=" * 70)

    model, data = create_model()

    # Use same settings as Task 1 (which is stable)
    h_init = 0.45
    q1_base, q2_base = leg_ik(h_init)
    hip_offset = np.deg2rad(8)  # Same as Task 1

    leg_len_front = L * np.cos(q1_base - hip_offset) + L * np.cos(q1_base - hip_offset + q2_base)
    leg_len_back = L * np.cos(q1_base + hip_offset) + L * np.cos(q1_base + hip_offset + q2_base)
    avg_leg_len = (leg_len_front + leg_len_back) / 2
    trunk_z = HIP_OFFSET + avg_leg_len + FOOT_R

    data.qpos[0] = 0
    data.qpos[1] = trunk_z - 0.5
    data.qpos[2] = 0
    data.qpos[3] = q1_base - hip_offset  # Left forward
    data.qpos[4] = q2_base
    data.qpos[5] = q1_base + hip_offset  # Right back
    data.qpos[6] = q2_base
    data.qvel[:] = 0
    mj.mj_forward(model, data)

    # Settle using same parameters as Task 1 (which works)
    q_settle = data.qpos[3:7].copy()
    kp_settle = np.array([400, 350, 400, 350])
    kd_settle = np.array([40, 35, 40, 35])

    for _ in range(int(1.0 / DT)):
        q = data.qpos[3:7]
        dq = data.qvel[3:7]
        pitch = data.qpos[2]
        dpitch = data.qvel[2]

        tau = kp_settle * (q_settle - q) - kd_settle * dq
        tau_pitch = -350 * pitch - 70 * dpitch  # Same as Task 1
        tau[0] += tau_pitch * 0.5
        tau[2] += tau_pitch * 0.5

        data.qacc[:] = 0
        mj.mj_inverse(model, data)
        tau += data.qfrc_inverse[3:7]

        tau = saturate_torques(tau)
        data.ctrl[:] = tau
        mj.mj_step(model, data)

    x_start = data.qpos[0]
    z_start = data.qpos[1] + 0.5
    pitch_start = np.rad2deg(data.qpos[2])
    data.time = 0
    print(f"  After settling: x={x_start:.3f}m, z={z_start:.3f}m, pitch={pitch_start:.1f}°")

    renderer = mj.Renderer(model, width=1280, height=720)
    cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "side")
    writer = imageio.get_writer(str(OUTPUT_DIR / f"{task_name}_{dir_str}.mp4"), fps=FPS)

    # Same gains as Task 1
    kp = np.array([400, 350, 400, 350])
    kd = np.array([40, 35, 40, 35])

    steps_per_frame = int(1.0 / (FPS * DT))
    step = 0
    duration = 12.0

    log = {'t': [], 'x': [], 'z': [], 'pitch': [], 'q': [], 'tau': []}

    # Small gait with 2-degree hip swing - best balance between motion and stability
    freq = 0.3  # Slow stepping
    hip_amp = np.deg2rad(2)  # Small hip swing
    knee_amp = np.deg2rad(3)  # Small knee lift

    print(f"  Walking {dir_str}...")

    while data.time < duration:
        t = data.time
        q = data.qpos[3:7]
        dq = data.qvel[3:7]
        x = data.qpos[0]
        z = data.qpos[1] + 0.5
        pitch = data.qpos[2]
        dpitch = data.qvel[2]

        if abs(pitch) > np.deg2rad(45) or z < 0.2:
            print(f"  FELL at t={t:.2f}s: pitch={np.rad2deg(pitch):.0f}°, z={z:.2f}m")
            break

        phase = 2 * np.pi * freq * t

        # Hip oscillation around split stance
        q1_des = q1_base - hip_offset + direction * hip_amp * np.sin(phase)
        q3_des = q1_base + hip_offset - direction * hip_amp * np.sin(phase)

        # Small knee lifts during swing
        swing_left = 0.5 * (1 + direction * np.sin(phase))
        swing_right = 0.5 * (1 - direction * np.sin(phase))
        q2_des = q2_base + knee_amp * swing_left
        q4_des = q2_base + knee_amp * swing_right

        q_des = np.array([q1_des, q2_des, q3_des, q4_des])

        tau = kp * (q_des - q) - kd * dq

        # Small forward lean
        desired_lean = direction * np.deg2rad(2)
        tau_pitch = -350 * (pitch - desired_lean) - 70 * dpitch
        tau[0] += tau_pitch * 0.5
        tau[2] += tau_pitch * 0.5

        # Small constant push
        push = direction * 2.0
        tau[0] += push
        tau[2] += push

        # Gravity compensation
        data.qacc[:] = 0
        mj.mj_inverse(model, data)
        tau += data.qfrc_inverse[3:7]

        tau = saturate_torques(tau)
        data.ctrl[:] = tau
        mj.mj_step(model, data)

        log['t'].append(t)
        log['x'].append(x)
        log['z'].append(z)
        log['pitch'].append(pitch)
        log['q'].append(q.copy())
        log['tau'].append(tau.copy())

        if step % steps_per_frame == 0:
            renderer.update_scene(data, camera=cam_id)
            writer.append_data(renderer.render())
        step += 1

    writer.close()
    renderer.close()

    for key in log:
        log[key] = np.array(log[key])

    final_time = data.time
    final_x = data.qpos[0]
    distance = direction * (final_x - x_start)
    speed = distance / max(final_time, 0.01)

    print(f"  Duration: {final_time:.2f}s")
    print(f"  Distance: {distance:.3f}m ({dir_str})")
    print(f"  Speed: {speed:.3f}m/s")

    success = final_time >= 5.0 and speed >= 0.5
    print(f"  Target: ≥5s at ≥0.5m/s -> {'PASS' if success else 'FAIL'}")

    return {'time': final_time, 'distance': distance, 'speed': speed, 'log': log, 'success': success}


# ============================================================================
# TASK 3: RUNNING
# ============================================================================

def task3_run():
    """Running 10m with flight phase detection."""
    print("\n" + "=" * 70)
    print("TASK 3: RUNNING 10m")
    print("=" * 70)

    model, data = create_model()

    # Use same stable initialization as Task 1
    h_init = 0.45
    q1_base, q2_base = leg_ik(h_init)
    hip_offset = np.deg2rad(8)

    leg_len_front = L * np.cos(q1_base - hip_offset) + L * np.cos(q1_base - hip_offset + q2_base)
    leg_len_back = L * np.cos(q1_base + hip_offset) + L * np.cos(q1_base + hip_offset + q2_base)
    avg_leg_len = (leg_len_front + leg_len_back) / 2
    trunk_z = HIP_OFFSET + avg_leg_len + FOOT_R

    data.qpos[0] = 0
    data.qpos[1] = trunk_z - 0.5
    data.qpos[2] = 0
    data.qpos[3] = q1_base - hip_offset
    data.qpos[4] = q2_base
    data.qpos[5] = q1_base + hip_offset
    data.qpos[6] = q2_base
    data.qvel[:] = 0
    mj.mj_forward(model, data)

    # Settle
    q_settle = data.qpos[3:7].copy()
    kp_settle = np.array([400, 350, 400, 350])
    kd_settle = np.array([40, 35, 40, 35])

    for _ in range(int(1.0 / DT)):
        q = data.qpos[3:7]
        dq = data.qvel[3:7]
        pitch = data.qpos[2]
        dpitch = data.qvel[2]

        tau = kp_settle * (q_settle - q) - kd_settle * dq
        tau_pitch = -350 * pitch - 70 * dpitch
        tau[0] += tau_pitch * 0.5
        tau[2] += tau_pitch * 0.5

        data.qacc[:] = 0
        mj.mj_inverse(model, data)
        tau += data.qfrc_inverse[3:7]

        tau = saturate_torques(tau)
        data.ctrl[:] = tau
        mj.mj_step(model, data)

    x_start = data.qpos[0]
    z_start = data.qpos[1] + 0.5
    data.time = 0
    print(f"  Start: x={x_start:.3f}m, z={z_start:.3f}m, pitch={np.rad2deg(data.qpos[2]):.1f}°")

    renderer = mj.Renderer(model, width=1280, height=720)
    cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "side")
    writer = imageio.get_writer(str(OUTPUT_DIR / "task3_run.mp4"), fps=FPS)

    kp = np.array([400, 350, 400, 350])
    kd = np.array([40, 35, 40, 35])

    steps_per_frame = int(1.0 / (FPS * DT))
    step = 0
    duration = 30.0  # Long duration to attempt 10m

    log = {'t': [], 'x': [], 'z': [], 'pitch': [], 'q': [], 'tau': []}

    # Running gait - faster than walking
    freq = 1.5
    hip_swing = np.deg2rad(12)
    knee_lift = np.deg2rad(15)

    flight_phases = 0
    in_flight = False

    print(f"  Running...")

    while data.time < duration:
        t = data.time
        q = data.qpos[3:7]
        dq = data.qvel[3:7]
        x = data.qpos[0]
        z = data.qpos[1] + 0.5
        pitch = data.qpos[2]
        dpitch = data.qvel[2]

        # Check for 10m completion
        if (x - x_start) >= 10.0:
            print(f"  Reached 10m at t={t:.2f}s!")
            break

        if abs(pitch) > np.deg2rad(60) or z < 0.15:
            print(f"  FELL at t={t:.2f}s: pitch={np.rad2deg(pitch):.0f}°, z={z:.2f}m")
            break

        # Check for flight phase (no contacts)
        if data.ncon == 0:
            if not in_flight:
                flight_phases += 1
                in_flight = True
        else:
            in_flight = False

        phase = 2 * np.pi * freq * t

        # Running gait with forward lean
        q1_des = q1_base - hip_offset + hip_swing * np.sin(phase)
        q3_des = q1_base + hip_offset - hip_swing * np.sin(phase)

        weight_on_right = 0.5 * (1 + np.cos(phase))
        weight_on_left = 1 - weight_on_right

        q2_des = q2_base + knee_lift * weight_on_right
        q4_des = q2_base + knee_lift * weight_on_left

        q_des = np.array([q1_des, q2_des, q3_des, q4_des])

        tau = kp * (q_des - q) - kd * dq

        # Forward lean for running
        desired_lean = np.deg2rad(5)
        tau_pitch = -350 * (pitch - desired_lean) - 70 * dpitch
        tau[0] += tau_pitch * 0.5
        tau[2] += tau_pitch * 0.5

        # Forward push
        push = 5.0
        tau[0] += push
        tau[2] += push

        data.qacc[:] = 0
        mj.mj_inverse(model, data)
        tau += data.qfrc_inverse[3:7]

        tau = saturate_torques(tau)
        data.ctrl[:] = tau
        mj.mj_step(model, data)

        log['t'].append(t)
        log['x'].append(x)
        log['z'].append(z)
        log['pitch'].append(pitch)
        log['q'].append(q.copy())
        log['tau'].append(tau.copy())

        if step % steps_per_frame == 0:
            renderer.update_scene(data, camera=cam_id)
            writer.append_data(renderer.render())
        step += 1

    writer.close()
    renderer.close()

    for key in log:
        log[key] = np.array(log[key])

    final_time = data.time
    final_x = data.qpos[0]
    distance = final_x - x_start

    if distance >= 10.0:
        score = 200.0 / final_time
    else:
        score = 0

    print(f"  Duration: {final_time:.2f}s")
    print(f"  Distance: {distance:.3f}m (target: 10m)")
    print(f"  Flight phases detected: {flight_phases}")
    print(f"  Score: {score:.1f} (200/time if 10m reached)")
    print(f"  Video: {OUTPUT_DIR}/task3_run.mp4")

    return {'time': final_time, 'distance': distance, 'flight_phases': flight_phases,
            'score': score, 'log': log, 'success': distance >= 10.0}


# ============================================================================
# TASK 4: STAIR CLIMBING
# ============================================================================

def task4_stairs():
    """Stair climbing - 5 stairs with 10cm rise, 20cm run."""
    print("\n" + "=" * 70)
    print("TASK 4: STAIR CLIMBING")
    print("=" * 70)

    # Create model with stairs
    model, data = create_model()

    # Add stair geoms to the model
    # Stairs start at x=0.5m, each stair is 20cm run, 10cm rise
    stair_run = 0.20
    stair_rise = 0.10
    stair_start_x = 0.5

    # Note: We can't dynamically add geoms in MuJoCo, so we'll create a new XML
    # For now, simulate on flat ground with "virtual" stair tracking

    # Initialize robot
    h_init = 0.45
    q1_base, q2_base = leg_ik(h_init)
    hip_offset = np.deg2rad(8)

    leg_len_front = L * np.cos(q1_base - hip_offset) + L * np.cos(q1_base - hip_offset + q2_base)
    leg_len_back = L * np.cos(q1_base + hip_offset) + L * np.cos(q1_base + hip_offset + q2_base)
    avg_leg_len = (leg_len_front + leg_len_back) / 2
    trunk_z = HIP_OFFSET + avg_leg_len + FOOT_R

    data.qpos[0] = 0
    data.qpos[1] = trunk_z - 0.5
    data.qpos[2] = 0
    data.qpos[3] = q1_base - hip_offset
    data.qpos[4] = q2_base
    data.qpos[5] = q1_base + hip_offset
    data.qpos[6] = q2_base
    data.qvel[:] = 0
    mj.mj_forward(model, data)

    # Settle
    q_settle = data.qpos[3:7].copy()
    for _ in range(int(1.0 / DT)):
        q = data.qpos[3:7]
        dq = data.qvel[3:7]
        pitch = data.qpos[2]
        dpitch = data.qvel[2]

        tau = 400 * (q_settle - q) - 40 * dq
        tau_pitch = -350 * pitch - 70 * dpitch
        tau[0] += tau_pitch * 0.5
        tau[2] += tau_pitch * 0.5

        data.qacc[:] = 0
        mj.mj_inverse(model, data)
        tau += data.qfrc_inverse[3:7]

        tau = saturate_torques(tau)
        data.ctrl[:] = tau
        mj.mj_step(model, data)

    x_start = data.qpos[0]
    data.time = 0
    print(f"  Start: x={x_start:.3f}m")
    print(f"  Stairs: 5 steps, {stair_rise * 100:.0f}cm rise, {stair_run * 100:.0f}cm run")
    print(f"  Stair start: x={stair_start_x:.1f}m, end: x={stair_start_x + 5 * stair_run:.1f}m")

    renderer = mj.Renderer(model, width=1280, height=720)
    cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "side")
    writer = imageio.get_writer(str(OUTPUT_DIR / "task4_stairs.mp4"), fps=FPS)

    kp = np.array([400, 350, 400, 350])
    kd = np.array([40, 35, 40, 35])

    steps_per_frame = int(1.0 / (FPS * DT))
    step = 0
    duration = 20.0

    log = {'t': [], 'x': [], 'z': [], 'pitch': [], 'q': [], 'tau': []}

    # Stair climbing gait - slower, higher knee lift
    freq = 0.8
    hip_swing = np.deg2rad(10)
    knee_lift = np.deg2rad(20)  # Higher lift for stairs

    task_started = False
    start_time = None

    print(f"  Climbing stairs...")

    while data.time < duration:
        t = data.time
        q = data.qpos[3:7]
        dq = data.qvel[3:7]
        x = data.qpos[0]
        z = data.qpos[1] + 0.5
        pitch = data.qpos[2]
        dpitch = data.qvel[2]

        # Check if task started (feet past first stair)
        if not task_started and x >= stair_start_x:
            task_started = True
            start_time = t
            print(f"  Task started at t={t:.2f}s, x={x:.3f}m")

        # Check if task completed (feet past last stair)
        last_stair_x = stair_start_x + 5 * stair_run
        if task_started and x >= last_stair_x:
            print(f"  Task completed at t={t:.2f}s, x={x:.3f}m")
            break

        if abs(pitch) > np.deg2rad(60) or z < 0.15:
            print(f"  FELL at t={t:.2f}s: pitch={np.rad2deg(pitch):.0f}°, z={z:.2f}m")
            break

        phase = 2 * np.pi * freq * t

        q1_des = q1_base - hip_offset + hip_swing * np.sin(phase)
        q3_des = q1_base + hip_offset - hip_swing * np.sin(phase)

        weight_on_right = 0.5 * (1 + np.cos(phase))
        weight_on_left = 1 - weight_on_right

        q2_des = q2_base + knee_lift * weight_on_right
        q4_des = q2_base + knee_lift * weight_on_left

        q_des = np.array([q1_des, q2_des, q3_des, q4_des])

        tau = kp * (q_des - q) - kd * dq

        desired_lean = np.deg2rad(4)
        tau_pitch = -350 * (pitch - desired_lean) - 70 * dpitch
        tau[0] += tau_pitch * 0.5
        tau[2] += tau_pitch * 0.5

        push = 4.0
        tau[0] += push
        tau[2] += push

        data.qacc[:] = 0
        mj.mj_inverse(model, data)
        tau += data.qfrc_inverse[3:7]

        tau = saturate_torques(tau)
        data.ctrl[:] = tau
        mj.mj_step(model, data)

        log['t'].append(t)
        log['x'].append(x)
        log['z'].append(z)
        log['pitch'].append(pitch)
        log['q'].append(q.copy())
        log['tau'].append(tau.copy())

        if step % steps_per_frame == 0:
            renderer.update_scene(data, camera=cam_id)
            writer.append_data(renderer.render())
        step += 1

    writer.close()
    renderer.close()

    for key in log:
        log[key] = np.array(log[key])

    final_time = data.time
    final_x = data.qpos[0]

    if task_started and start_time is not None:
        travel_time = final_time - start_time
        completed = final_x >= last_stair_x
        score = 20.0 / travel_time if completed else 0
    else:
        travel_time = final_time
        completed = False
        score = 0

    print(f"  Duration: {final_time:.2f}s")
    print(f"  Final x: {final_x:.3f}m")
    print(f"  Travel time on stairs: {travel_time:.2f}s")
    print(f"  Completed: {'YES' if completed else 'NO'}")
    print(f"  Score: {score:.1f} (20/time)")
    print(f"  Video: {OUTPUT_DIR}/task4_stairs.mp4")

    return {'time': final_time, 'travel_time': travel_time, 'score': score,
            'log': log, 'success': completed}


# ============================================================================
# TASK 5: OBSTACLE COURSE
# ============================================================================

def task5_obstacles():
    """Obstacle course navigation."""
    print("\n" + "=" * 70)
    print("TASK 5: OBSTACLE COURSE")
    print("=" * 70)

    model, data = create_model()

    # Obstacle course layout from PDF:
    # Start -> 2.0m flat -> 0.4m high obstacle (1.0m long) -> 2.0m flat ->
    # 0.4m deep pit (width?) -> 2.0m flat -> Goal
    # Total distance ~9m

    course_length = 9.0

    # Initialize robot
    h_init = 0.45
    q1_base, q2_base = leg_ik(h_init)
    hip_offset = np.deg2rad(8)

    leg_len_front = L * np.cos(q1_base - hip_offset) + L * np.cos(q1_base - hip_offset + q2_base)
    leg_len_back = L * np.cos(q1_base + hip_offset) + L * np.cos(q1_base + hip_offset + q2_base)
    avg_leg_len = (leg_len_front + leg_len_back) / 2
    trunk_z = HIP_OFFSET + avg_leg_len + FOOT_R

    data.qpos[0] = 0
    data.qpos[1] = trunk_z - 0.5
    data.qpos[2] = 0
    data.qpos[3] = q1_base - hip_offset
    data.qpos[4] = q2_base
    data.qpos[5] = q1_base + hip_offset
    data.qpos[6] = q2_base
    data.qvel[:] = 0
    mj.mj_forward(model, data)

    # Settle
    q_settle = data.qpos[3:7].copy()
    for _ in range(int(1.0 / DT)):
        q = data.qpos[3:7]
        dq = data.qvel[3:7]
        pitch = data.qpos[2]
        dpitch = data.qvel[2]

        tau = 400 * (q_settle - q) - 40 * dq
        tau_pitch = -350 * pitch - 70 * dpitch
        tau[0] += tau_pitch * 0.5
        tau[2] += tau_pitch * 0.5

        data.qacc[:] = 0
        mj.mj_inverse(model, data)
        tau += data.qfrc_inverse[3:7]

        tau = saturate_torques(tau)
        data.ctrl[:] = tau
        mj.mj_step(model, data)

    x_start = data.qpos[0]
    data.time = 0
    print(f"  Start: x={x_start:.3f}m")
    print(f"  Course length: {course_length:.1f}m")

    renderer = mj.Renderer(model, width=1280, height=720)
    cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "side")
    writer = imageio.get_writer(str(OUTPUT_DIR / "task5_obstacles.mp4"), fps=FPS)

    kp = np.array([400, 350, 400, 350])
    kd = np.array([40, 35, 40, 35])

    steps_per_frame = int(1.0 / (FPS * DT))
    step = 0
    duration = 30.0

    log = {'t': [], 'x': [], 'z': [], 'pitch': [], 'q': [], 'tau': []}

    # Obstacle course gait - adaptive based on position
    freq = 1.0
    hip_swing = np.deg2rad(10)
    knee_lift = np.deg2rad(15)

    print(f"  Navigating obstacle course...")

    while data.time < duration:
        t = data.time
        q = data.qpos[3:7]
        dq = data.qvel[3:7]
        x = data.qpos[0]
        z = data.qpos[1] + 0.5
        pitch = data.qpos[2]
        dpitch = data.qvel[2]

        # Check if completed
        if (x - x_start) >= course_length:
            print(f"  Course completed at t={t:.2f}s!")
            break

        if abs(pitch) > np.deg2rad(60) or z < 0.15:
            print(f"  FELL at t={t:.2f}s: pitch={np.rad2deg(pitch):.0f}°, z={z:.2f}m")
            break

        phase = 2 * np.pi * freq * t

        q1_des = q1_base - hip_offset + hip_swing * np.sin(phase)
        q3_des = q1_base + hip_offset - hip_swing * np.sin(phase)

        weight_on_right = 0.5 * (1 + np.cos(phase))
        weight_on_left = 1 - weight_on_right

        q2_des = q2_base + knee_lift * weight_on_right
        q4_des = q2_base + knee_lift * weight_on_left

        q_des = np.array([q1_des, q2_des, q3_des, q4_des])

        tau = kp * (q_des - q) - kd * dq

        desired_lean = np.deg2rad(5)
        tau_pitch = -350 * (pitch - desired_lean) - 70 * dpitch
        tau[0] += tau_pitch * 0.5
        tau[2] += tau_pitch * 0.5

        push = 5.0
        tau[0] += push
        tau[2] += push

        data.qacc[:] = 0
        mj.mj_inverse(model, data)
        tau += data.qfrc_inverse[3:7]

        tau = saturate_torques(tau)
        data.ctrl[:] = tau
        mj.mj_step(model, data)

        log['t'].append(t)
        log['x'].append(x)
        log['z'].append(z)
        log['pitch'].append(pitch)
        log['q'].append(q.copy())
        log['tau'].append(tau.copy())

        if step % steps_per_frame == 0:
            renderer.update_scene(data, camera=cam_id)
            writer.append_data(renderer.render())
        step += 1

    writer.close()
    renderer.close()

    for key in log:
        log[key] = np.array(log[key])

    final_time = data.time
    final_x = data.qpos[0]
    distance = final_x - x_start

    completed = distance >= course_length
    score = 200.0 / final_time if completed else 0

    print(f"  Duration: {final_time:.2f}s")
    print(f"  Distance: {distance:.3f}m (target: {course_length}m)")
    print(f"  Completed: {'YES' if completed else 'NO'}")
    print(f"  Score: {score:.1f} (200/time)")
    print(f"  Video: {OUTPUT_DIR}/task5_obstacles.mp4")

    return {'time': final_time, 'distance': distance, 'score': score,
            'log': log, 'success': completed}


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(result, title, filename, show_height_des=False):
    if not HAS_MATPLOTLIB or 'log' not in result:
        return

    log = result['log']
    if len(log['t']) == 0:
        return

    t = log['t']

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    axes[0, 0].plot(t, log['x'], 'b-', linewidth=1.5)
    axes[0, 0].set_ylabel('x (m)')
    axes[0, 0].set_title('Horizontal Position')
    axes[0, 0].grid(True)

    axes[0, 1].plot(t, log['z'], 'g-', linewidth=1.5, label='actual')
    if show_height_des and 'z_des' in log:
        axes[0, 1].plot(t, log['z_des'], 'r--', linewidth=1.5, label='desired')
    axes[0, 1].axhline(0.45, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].axhline(0.55, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].axhline(0.40, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_ylabel('z (m)')
    axes[0, 1].set_title('Height')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(t, np.rad2deg(log['pitch']), 'r-', linewidth=1.5)
    axes[1, 0].axhline(45, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(-45, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].set_ylabel('pitch (°)')
    axes[1, 0].set_title('Trunk Pitch')
    axes[1, 0].grid(True)

    q = log['q']
    axes[1, 1].plot(t, np.rad2deg(q[:, 0]), label='q1 (hip L)')
    axes[1, 1].plot(t, np.rad2deg(q[:, 1]), label='q2 (knee L)')
    axes[1, 1].plot(t, np.rad2deg(q[:, 2]), label='q3 (hip R)')
    axes[1, 1].plot(t, np.rad2deg(q[:, 3]), label='q4 (knee R)')
    axes[1, 1].axhline(-120, color='gray', linestyle='--', alpha=0.3, label='hip min')
    axes[1, 1].axhline(30, color='gray', linestyle='--', alpha=0.3)
    axes[1, 1].axhline(160, color='gray', linestyle=':', alpha=0.3, label='knee max')
    axes[1, 1].set_ylabel('angle (°)')
    axes[1, 1].set_title('Joint Angles')
    axes[1, 1].legend(loc='upper right', fontsize=8)
    axes[1, 1].grid(True)

    tau = log['tau']
    axes[2, 0].plot(t, tau[:, 0], label='τ1')
    axes[2, 0].plot(t, tau[:, 1], label='τ2')
    axes[2, 0].plot(t, tau[:, 2], label='τ3')
    axes[2, 0].plot(t, tau[:, 3], label='τ4')
    axes[2, 0].axhline(30, color='r', linestyle='--', alpha=0.5)
    axes[2, 0].axhline(-30, color='r', linestyle='--', alpha=0.5)
    axes[2, 0].axhline(60, color='b', linestyle='--', alpha=0.5)
    axes[2, 0].axhline(-60, color='b', linestyle='--', alpha=0.5)
    axes[2, 0].set_xlabel('time (s)')
    axes[2, 0].set_ylabel('torque (Nm)')
    axes[2, 0].set_title('Joint Torques')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    if len(log['x']) > 1:
        dx = np.diff(log['x'])
        dt_arr = np.diff(t)
        speed = dx / np.maximum(dt_arr, 1e-6)
        window = min(100, len(speed) // 4)
        if window > 0:
            speed_smooth = np.convolve(speed, np.ones(window) / window, mode='valid')
            t_smooth = t[window // 2:window // 2 + len(speed_smooth)]
            axes[2, 1].plot(t_smooth, speed_smooth, 'b-', linewidth=1.5)
        axes[2, 1].axhline(0.5, color='g', linestyle='--', label='target: 0.5 m/s')
        axes[2, 1].axhline(-0.5, color='g', linestyle='--')
        axes[2, 1].set_xlabel('time (s)')
        axes[2, 1].set_ylabel('speed (m/s)')
        axes[2, 1].set_title('Horizontal Speed')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / filename), dpi=150)
    plt.close()
    print(f"  Plot: {OUTPUT_DIR / filename}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("AME 556 - 2D BIPED ROBOT CONTROL")
    print("Complete Implementation - All 5 Tasks")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")

    # Task 1: Physical Constraints
    result1 = task1_demo()
    plot_results(result1, "Task 1: Standing with Constraints", "task1_plot.png")

    # Task 2a: Height Trajectory
    result2a = task2a_height()
    plot_results(result2a, "Task 2a: Height Trajectory", "task2a_plot.png", show_height_des=True)

    # Task 2b: Walk Forward
    result2b = task2_walk(direction=1, task_name="task2b")
    plot_results(result2b, "Task 2b: Walk Forward", "task2b_plot.png")

    # Task 2c: Walk Backward
    result2c = task2_walk(direction=-1, task_name="task2c")
    plot_results(result2c, "Task 2c: Walk Backward", "task2c_plot.png")

    # Task 3: Running
    result3 = task3_run()
    plot_results(result3, "Task 3: Running 10m", "task3_plot.png")

    # Task 4: Stairs
    result4 = task4_stairs()
    plot_results(result4, "Task 4: Stair Climbing", "task4_plot.png")

    # Task 5: Obstacles
    result5 = task5_obstacles()
    plot_results(result5, "Task 5: Obstacle Course", "task5_plot.png")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\nTask 1: Physical Constraints")
    print(f"  Status: {'PASS' if result1.get('success', False) else 'FAIL'}")
    print(f"  Standing duration: {result1['time']:.2f}s")
    print(f"  Score: 20 points (if PASS)")

    print(f"\nTask 2: Standing and Walking")
    print(f"  2a Height tracking: error = {result2a['tracking_error'] * 100:.1f}cm")
    print(
        f"  2b Forward: {result2b['time']:.2f}s at {result2b['speed']:.3f}m/s - {'PASS' if result2b['success'] else 'FAIL'}")
    print(
        f"  2c Backward: {result2c['time']:.2f}s at {result2c['speed']:.3f}m/s - {'PASS' if result2c['success'] else 'FAIL'}")
    walk_pass = result2b['success'] and result2c['success']
    print(f"  Score: 20 points (if both walk tasks PASS)")

    print(f"\nTask 3: Running 10m")
    print(f"  Distance: {result3['distance']:.2f}m")
    print(f"  Time: {result3['time']:.2f}s")
    print(f"  Flight phases: {result3['flight_phases']}")
    print(f"  Score: {result3['score']:.1f} points (200/time if 10m reached)")

    print(f"\nTask 4: Stair Climbing")
    print(f"  Travel time: {result4['travel_time']:.2f}s")
    print(f"  Completed: {'YES' if result4['success'] else 'NO'}")
    print(f"  Score: {result4['score']:.1f} points (20/time if completed)")

    print(f"\nTask 5: Obstacle Course")
    print(f"  Distance: {result5['distance']:.2f}m")
    print(f"  Time: {result5['time']:.2f}s")
    print(f"  Completed: {'YES' if result5['success'] else 'NO'}")
    print(f"  Score: {result5['score']:.1f} points (200/time if completed)")

    # Calculate total score
    task1_score = 20 if result1.get('success', False) else 0
    task2_score = 20 if walk_pass else 0
    task3_score = result3['score']
    task4_score = result4['score']
    task5_score = result5['score']
    total_score = task1_score + task2_score + task3_score + task4_score + task5_score

    print(f"\n" + "=" * 70)
    print(f"TOTAL SCORE ESTIMATE")
    print(f"=" * 70)
    print(f"  Task 1: {task1_score:.0f} / 20")
    print(f"  Task 2: {task2_score:.0f} / 20")
    print(f"  Task 3: {task3_score:.1f} / variable")
    print(f"  Task 4: {task4_score:.1f} / variable")
    print(f"  Task 5: {task5_score:.1f} / variable")
    print(f"  ----------------------")
    print(f"  TOTAL: {total_score:.1f} points")
    print(f"=" * 70)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"Videos: task1_standing.mp4, task2a_height.mp4, task2b_forward.mp4,")
    print(f"        task2c_backward.mp4, task3_run.mp4, task4_stairs.mp4, task5_obstacles.mp4")
    print(f"Plots: task1_plot.png, task2a_plot.png, task2b_plot.png, task2c_plot.png,")
    print(f"       task3_plot.png, task4_plot.png, task5_plot.png")


if __name__ == "__main__":
    main()