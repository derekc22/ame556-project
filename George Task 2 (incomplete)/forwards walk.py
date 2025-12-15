#!/usr/bin/env python3
"""
FORWARD WALKING - Task 2b - BEST VERSION
Parameters: freq=1.2Hz, amp=11°, lean=-8°, Kp=70, Kd=17
Results: 10s, 2.07m, 0.207 m/s forward
"""

import os
import platform

if platform.system() == 'Linux':
    os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import mujoco as mj
import imageio
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HIP_ANGLE_MIN, HIP_ANGLE_MAX = -2.094, 0.524
KNEE_ANGLE_MIN, KNEE_ANGLE_MAX = 0.0, 2.793
HIP_VEL_MAX, KNEE_VEL_MAX = 30.0, 15.0
HIP_TAU_MAX, KNEE_TAU_MAX = 30.0, 60.0
L, HIP_OFFSET, FOOT_R, TRUNK_Z0 = 0.22, 0.125, 0.02, 0.65


def foot_positions_fk(q1, q2, q3, q4):
    lf_x = L * np.sin(q1) + L * np.sin(q1 + q2)
    lf_z = -L * np.cos(q1) - L * np.cos(q1 + q2)
    rf_x = L * np.sin(q3) + L * np.sin(q3 + q4)
    rf_z = -L * np.cos(q3) - L * np.cos(q3 + q4)
    return lf_x, lf_z, rf_x, rf_z


def gravity_compensation(model, data):
    qacc_save = data.qacc.copy()
    data.qacc[:] = 0
    mj.mj_inverse(model, data)
    tau_g = data.qfrc_inverse[3:7].copy()
    data.qacc[:] = qacc_save
    return tau_g


def saturate_torques(tau):
    tau = tau.copy()
    tau[0] = np.clip(tau[0], -HIP_TAU_MAX, HIP_TAU_MAX)
    tau[1] = np.clip(tau[1], -KNEE_TAU_MAX, KNEE_TAU_MAX)
    tau[2] = np.clip(tau[2], -HIP_TAU_MAX, HIP_TAU_MAX)
    tau[3] = np.clip(tau[3], -KNEE_TAU_MAX, KNEE_TAU_MAX)
    return tau


def check_constraints(q, dq):
    if q[0] < HIP_ANGLE_MIN: return True, f"q1={np.rad2deg(q[0]):.1f}°<-120°"
    if q[0] > HIP_ANGLE_MAX: return True, f"q1={np.rad2deg(q[0]):.1f}°>30°"
    if q[2] < HIP_ANGLE_MIN: return True, f"q3={np.rad2deg(q[2]):.1f}°<-120°"
    if q[2] > HIP_ANGLE_MAX: return True, f"q3={np.rad2deg(q[2]):.1f}°>30°"
    if q[1] < KNEE_ANGLE_MIN: return True, f"q2={np.rad2deg(q[1]):.1f}°<0°"
    if q[1] > KNEE_ANGLE_MAX: return True, f"q2={np.rad2deg(q[1]):.1f}°>160°"
    if q[3] < KNEE_ANGLE_MIN: return True, f"q4={np.rad2deg(q[3]):.1f}°<0°"
    if q[3] > KNEE_ANGLE_MAX: return True, f"q4={np.rad2deg(q[3]):.1f}°>160°"
    if abs(dq[0]) > HIP_VEL_MAX: return True, f"|dq1|={abs(dq[0]):.1f}>30"
    if abs(dq[2]) > HIP_VEL_MAX: return True, f"|dq3|={abs(dq[2]):.1f}>30"
    if abs(dq[1]) > KNEE_VEL_MAX: return True, f"|dq2|={abs(dq[1]):.1f}>15"
    if abs(dq[3]) > KNEE_VEL_MAX: return True, f"|dq4|={abs(dq[3]):.1f}>15"
    return False, ""


def run_forward_walk(xml_path, duration=10.0, video_name="walk_forward_best.mp4"):
    print(f"\n{'=' * 60}")
    print("FORWARD WALKING - Task 2b - BEST VERSION")
    print("Params: freq=1.2Hz, amp=11°, lean=-8°, Kp=70, Kd=17")
    print("=" * 60)

    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    mj.mj_resetData(model, data)
    q1_init, q2_init = np.deg2rad(-50), np.deg2rad(40)
    q3_init, q4_init = np.deg2rad(-5), np.deg2rad(55)
    data.qpos[3:7] = [q1_init, q2_init, q3_init, q4_init]

    lf_x, lf_z, rf_x, rf_z = foot_positions_fk(q1_init, q2_init, q3_init, q4_init)
    data.qpos[0] = -(lf_x + rf_x) / 2
    data.qpos[1] = HIP_OFFSET - min(lf_z, rf_z) + FOOT_R + 0.002 - TRUNK_Z0
    data.qpos[2] = 0.0
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    q0 = data.qpos[3:7].copy()

    Kp = np.array([200.0, 150.0, 200.0, 150.0])
    Kd = np.array([20.0, 15.0, 20.0, 15.0])

    for _ in range(300):
        q, dq = data.qpos[3:7], data.qvel[3:7]
        pitch, dpitch = data.qpos[2], data.qvel[2]
        tau = gravity_compensation(model, data) + Kp * (q0 - q) - Kd * dq
        tau[0] += 50 * (0 - pitch) - 12 * dpitch
        tau[2] += 50 * (0 - pitch) - 12 * dpitch
        data.ctrl[:] = saturate_torques(tau)
        mj.mj_step(model, data)

    data.qvel[:] = 0
    mj.mj_forward(model, data)
    x0 = data.qpos[0]
    q0 = data.qpos[3:7].copy()

    # BEST forward walking params
    freq = 1.2
    amp = np.deg2rad(11)
    lean = np.deg2rad(-8)
    pitch_kp, pitch_kd = 70.0, 17.0

    video_path = OUTPUT_DIR / video_name
    renderer = mj.Renderer(model, width=640, height=480)
    writer = imageio.get_writer(str(video_path), fps=60)
    dt = model.opt.timestep
    steps_per_frame = max(1, int(1.0 / (60 * dt)))

    step = 0
    termination_reason = None

    while data.time < duration:
        t = data.time
        q = data.qpos[3:7].copy()
        dq = data.qvel[3:7].copy()
        pitch, dpitch = data.qpos[2], data.qvel[2]

        violated, msg = check_constraints(q, dq)
        if violated:
            termination_reason = msg
            print(f"  t={t:.2f}s: CONSTRAINT: {msg}")
            break

        if abs(pitch) > np.deg2rad(60):
            termination_reason = f"FELL (pitch={np.rad2deg(pitch):.1f}°)"
            print(f"  t={t:.2f}s: {termination_reason}")
            break

        phase = 2 * np.pi * freq * t
        q_des = q0.copy()
        # FLIPPED hip oscillation for forward motion
        q_des[0] = q0[0] + lean - amp * np.sin(phase)
        q_des[2] = q0[2] + lean + amp * np.sin(phase)
        # No knee lift

        tau = gravity_compensation(model, data) + Kp * (q_des - q) - Kd * dq
        tau[0] += pitch_kp * (0 - pitch) - pitch_kd * dpitch
        tau[2] += pitch_kp * (0 - pitch) - pitch_kd * dpitch

        data.ctrl[:] = saturate_torques(tau)
        mj.mj_step(model, data)

        if step % steps_per_frame == 0:
            renderer.update_scene(data)
            writer.append_data(renderer.render())

        if step % int(1.0 / dt) == 0:
            print(f"  t={t:.1f}s: x={data.qpos[0]:.3f}m, v={data.qvel[0]:.3f}m/s")

        step += 1

    writer.close()
    renderer.close()

    final_t = data.time
    distance = data.qpos[0] - x0
    speed = abs(distance) / max(final_t, 0.01)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    print(f"Duration: {final_t:.2f}s")
    print(f"Distance: {distance:.3f}m (positive = forward)")
    print(f"Speed: {speed:.3f}m/s")
    if termination_reason:
        print(f"Terminated: {termination_reason}")
    print(f"\nTask 2b Requirements:")
    print(f"  Duration >= 5s: {final_t:.2f}s {'✓' if final_t >= 5 else '✗'}")
    print(f"  Speed >= 0.5m/s: {speed:.3f}m/s {'✓' if speed >= 0.5 else '✗'}")
    print(f"  Speed >= 0.2m/s: {speed:.3f}m/s {'✓' if speed >= 0.2 else '✗'}")

    return {'time': final_t, 'distance': distance, 'speed': speed, 'video': str(video_path)}


if __name__ == "__main__":
    xml_path = str(SCRIPT_DIR / "biped_robot.xml")
    result = run_forward_walk(xml_path, 10.0, "walk_forward_best.mp4")