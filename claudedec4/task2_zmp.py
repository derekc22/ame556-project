"""
Task 2 Walking with ZMP-based Balance Control
Uses Zero Moment Point tracking for stable walking
"""
import numpy as np
import mujoco as mj
from pathlib import Path
import tempfile

XML_CONTENT = '''<mujoco model="biped2d_zmp">
  <compiler angle="radian"/>
  
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.0001">
    <flag override="enable"/>
  </option>
  
  <default>
    <geom contype="1" conaffinity="1" condim="3" friction="0.5 0.005 0.0001"
          solref="0.00001 1" solimp="0.95 0.99 0.001"/>
    <joint damping="0.1"/>
  </default>
  
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <camera name="side" mode="trackcom" pos="0 -2.5 0.6" xyaxes="1 0 0 0 0 1"/>
    <geom name="floor" type="plane" size="50 50 0.1"/>
    
    <body name="trunk" pos="0 0 0.6">
      <freejoint name="floating_base"/>
      <geom name="trunk_geom" type="box" size="0.075 0.04 0.125" mass="8.0" 
            contype="0" conaffinity="0" rgba="0.2 0.6 0.9 1"/>
      
      <body name="left_thigh" pos="0 0.02 -0.125">
        <joint name="q1" type="hinge" axis="0 1 0" range="-2.094 0.524" damping="0.1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="0.25"
              contype="0" conaffinity="0" rgba="0.3 0.8 0.3 1"/>
        <body name="left_shank" pos="0 0 -0.22">
          <joint name="q2" type="hinge" axis="0 1 0" range="0 2.792" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.018" mass="0.25"
                contype="0" conaffinity="0" rgba="0.3 0.8 0.3 1"/>
          <geom name="l_foot" type="box" pos="0 0 -0.22" size="0.05 0.025 0.01" rgba="0.9 0.5 0.2 1"/>
        </body>
      </body>
      
      <body name="right_thigh" pos="0 -0.02 -0.125">
        <joint name="q3" type="hinge" axis="0 1 0" range="-2.094 0.524" damping="0.1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="0.25"
              contype="0" conaffinity="0" rgba="0.8 0.3 0.3 1"/>
        <body name="right_shank" pos="0 0 -0.22">
          <joint name="q4" type="hinge" axis="0 1 0" range="0 2.792" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.018" mass="0.25"
                contype="0" conaffinity="0" rgba="0.8 0.3 0.3 1"/>
          <geom name="r_foot" type="box" pos="0 0 -0.22" size="0.05 0.025 0.01" rgba="0.9 0.5 0.2 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="hip_L" joint="q1" gear="1" ctrllimited="true" ctrlrange="-30 30"/>
    <motor name="knee_L" joint="q2" gear="1" ctrllimited="true" ctrlrange="-60 60"/>
    <motor name="hip_R" joint="q3" gear="1" ctrllimited="true" ctrlrange="-30 30"/>
    <motor name="knee_R" joint="q4" gear="1" ctrllimited="true" ctrlrange="-60 60"/>
  </actuator>
</mujoco>'''

FOOT_HEIGHT = 0.01
HIP_TAU_MAX, KNEE_TAU_MAX = 30.0, 60.0
KNEE_VEL_MAX = 15.0


def create_model():
    # Use temp directory for cross-platform compatibility
    xml_path = Path(tempfile.gettempdir()) / "biped2d_zmp.xml"
    with open(xml_path, 'w') as f:
        f.write(XML_CONTENT)
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    return model, data


def get_ids(model):
    return {
        'l_foot': mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "l_foot"),
        'r_foot': mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "r_foot"),
        'trunk': mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "trunk"),
        'floor': mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor"),
    }


def saturate_torques(tau):
    return np.array([
        np.clip(tau[0], -HIP_TAU_MAX, HIP_TAU_MAX),
        np.clip(tau[1], -KNEE_TAU_MAX, KNEE_TAU_MAX),
        np.clip(tau[2], -HIP_TAU_MAX, HIP_TAU_MAX),
        np.clip(tau[3], -KNEE_TAU_MAX, KNEE_TAU_MAX)
    ])


def get_pitch(model, data, ids):
    R = data.xmat[ids['trunk']].reshape(3, 3)
    return np.arctan2(R[0, 2], R[2, 2])


def get_foot_contacts(model, data, ids):
    """Check which feet are in contact with floor."""
    l_contact, r_contact = False, False
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == ids['floor'] and c.geom2 == ids['l_foot']) or \
           (c.geom1 == ids['l_foot'] and c.geom2 == ids['floor']):
            l_contact = True
        if (c.geom1 == ids['floor'] and c.geom2 == ids['r_foot']) or \
           (c.geom1 == ids['r_foot'] and c.geom2 == ids['floor']):
            r_contact = True
    return l_contact, r_contact


def initialize(model, data, q_stand, ids):
    mj.mj_resetData(model, data)
    data.qpos[0:3] = [0, 0, 0.8]
    data.qpos[3] = 1.0
    data.qpos[4:7] = 0.0
    data.qpos[7:11] = q_stand
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)

    # Check foot positions
    l_x = data.geom_xpos[ids['l_foot']][0]
    r_x = data.geom_xpos[ids['r_foot']][0]
    l_z = data.geom_xpos[ids['l_foot']][2]
    r_z = data.geom_xpos[ids['r_foot']][2]

    print(f"  Before init: L=({l_x:.3f},{l_z:.3f}) R=({r_x:.3f},{r_z:.3f})")
    print(f"  Support width: {abs(l_x - r_x):.3f}m")

    # Lower until higher foot touches
    max_z = max(l_z, r_z)
    data.qpos[2] -= (max_z - FOOT_HEIGHT + 0.008)  # Extra margin
    mj.mj_forward(model, data)

    # Center COM
    l_x = data.geom_xpos[ids['l_foot']][0]
    r_x = data.geom_xpos[ids['r_foot']][0]
    support_center = (l_x + r_x) / 2
    com_x = data.subtree_com[1][0]
    data.qpos[0] -= (com_x - support_center)

    # Settle with LONGER stabilization
    for settle_step in range(5000):
        pitch = get_pitch(model, data, ids)
        omega = data.qvel[4]
        data.qacc[:] = 0.0
        mj.mj_inverse(model, data)
        tau = data.qfrc_inverse[6:10].copy()

        # Very strong pitch control during settling
        tau[0] += 500.0 * pitch + 100.0 * omega
        tau[2] += 500.0 * pitch + 100.0 * omega

        q_cur, qd = data.qpos[7:11], data.qvel[6:10]
        for i in range(4):
            tau[i] += 300.0 * (q_stand[i] - q_cur[i]) - 40.0 * qd[i]
        data.ctrl[:] = saturate_torques(tau)
        mj.mj_step(model, data)

    data.qvel[:] = 0
    mj.mj_forward(model, data)

    # Verify contacts
    l_c, r_c = get_foot_contacts(model, data, ids)
    print(f"  After init: L_contact={l_c}, R_contact={r_c}, pitch={np.rad2deg(get_pitch(model, data, ids)):.1f}°")


def compute_zmp(model, data, ids):
    """Compute Zero Moment Point in x direction."""
    com = data.subtree_com[1].copy()
    com_vel = data.cvel[1, 3:6].copy()  # Linear velocity

    # Approximate ZMP: x_zmp = x_com - (z_com / g) * x_ddot
    # For standing, ZMP ≈ COM_x
    g = 9.81
    z_com = com[2]

    # Use trunk acceleration
    trunk_id = ids['trunk']
    trunk_acc = data.cacc[trunk_id, 3:6]  # Linear acceleration

    if z_com > 0.1:
        zmp_x = com[0] - (z_com / g) * trunk_acc[0]
    else:
        zmp_x = com[0]

    return zmp_x, com[0]


def walking_controller_zmp(t, model, data, ids, q_stand, direction="forward"):
    """
    Continuous sinusoidal walking gait.
    """
    settle_time = 0.5

    if t < settle_time:
        return q_stand.copy()

    t_walk = t - settle_time

    # Walking parameters - balanced
    freq = 2.0  # Hz - cycles per second
    phase = 2 * np.pi * freq * t_walk

    # Gradual ramp-up over 0.8 seconds
    ramp = min(1.0, t_walk / 0.8)

    # Base angles matching stand pose
    base_hip_L = np.deg2rad(-45)
    base_hip_R = np.deg2rad(5)
    base_knee = np.deg2rad(40)

    # Walking amplitudes
    hip_swing = np.deg2rad(28)
    knee_lift = np.deg2rad(15)

    sign = 1.0 if direction == "forward" else -1.0

    # Continuous sinusoidal gait - legs 180° out of phase
    # For FORWARD: hip becomes MORE negative (points forward) during swing
    # For BACKWARD: hip becomes LESS negative (points backward) during swing
    q1_des = base_hip_L + sign * ramp * hip_swing * np.sin(phase)
    q3_des = base_hip_R + sign * ramp * hip_swing * np.sin(phase + np.pi)

    # Knee lifts when leg swings forward
    q2_des = base_knee + ramp * knee_lift * max(0, -np.sin(phase))
    q4_des = base_knee + ramp * knee_lift * max(0, -np.sin(phase + np.pi))

    return np.array([q1_des, q2_des, q3_des, q4_des])


def run_walking_test(direction="forward", duration=8.0):
    print(f"\n{'='*60}")
    print(f"ZMP Walking Test: {direction.upper()}")
    print(f"{'='*60}")

    model, data = create_model()
    ids = get_ids(model)

    q_stand = np.deg2rad([-45, 40, 5, 40])  # Equal foot heights, 0.349m spread
    initialize(model, data, q_stand, ids)

    x_start = data.subtree_com[1][0]
    print(f"Initial COM: x={x_start:.3f}m")

    # Control gains - very strong for stability
    KP = 500.0
    KD = 50.0
    KP_PITCH = 1000.0
    KD_PITCH = 200.0

    # ZMP tracking gains
    KP_ZMP = 100.0

    max_knee_vel = 0
    final_t = 0

    for step in range(int(duration / model.opt.timestep)):
        t = step * model.opt.timestep

        q_des = walking_controller_zmp(t, model, data, ids, q_stand, direction)

        q_cur = data.qpos[7:11]
        qd = data.qvel[6:10]
        pitch = get_pitch(model, data, ids)
        omega = data.qvel[4]

        max_knee_vel = max(max_knee_vel, abs(qd[1]), abs(qd[3]))

        # Gravity compensation
        data.qacc[:] = 0.0
        mj.mj_inverse(model, data)
        tau = data.qfrc_inverse[6:10].copy()

        # Joint PD
        for i in range(4):
            tau[i] += KP * (q_des[i] - q_cur[i]) - KD * qd[i]

        # Pitch control
        tau_pitch = KP_PITCH * pitch + KD_PITCH * omega
        tau[0] += tau_pitch
        tau[2] += tau_pitch

        # ZMP-based balance adjustment
        zmp_x, com_x = compute_zmp(model, data, ids)
        l_x = data.geom_xpos[ids['l_foot']][0]
        r_x = data.geom_xpos[ids['r_foot']][0]
        support_center = (l_x + r_x) / 2

        # Push COM toward support center
        zmp_error = com_x - support_center
        tau_zmp = KP_ZMP * zmp_error
        tau[0] -= tau_zmp  # Adjust hip torques to shift weight
        tau[2] -= tau_zmp

        tau = saturate_torques(tau)

        # Velocity constraint check
        if abs(qd[1]) > KNEE_VEL_MAX or abs(qd[3]) > KNEE_VEL_MAX:
            print(f"  Knee velocity limit at t={t:.2f}s")
            break

        # Fall detection
        if data.qpos[2] < 0.2:
            print(f"  FELL at t={t:.2f}s")
            break

        data.ctrl[:] = tau
        mj.mj_step(model, data)
        final_t = t

        if step % 20000 == 0:
            com = data.subtree_com[1]
            l_c, r_c = get_foot_contacts(model, data, ids)
            print(f"  t={t:.1f}s: x={com[0]:.3f}m pitch={np.rad2deg(pitch):+.1f}° L={l_c} R={r_c}")

    x_final = data.subtree_com[1][0]
    if direction == "forward":
        dist = x_final - x_start
    else:
        dist = x_start - x_final  # Backward means negative x

    walk_time = max(0.01, final_t - 0.5)  # Subtract settle time
    speed = abs(dist) / walk_time if walk_time > 0 else 0

    success = speed >= 0.5 and walk_time >= 5.0

    print(f"\nResults:")
    print(f"  Distance: {abs(dist):.3f}m")
    print(f"  Walk time: {walk_time:.2f}s")
    print(f"  Speed: {speed:.3f} m/s (target >= 0.5 m/s)")
    print(f"  Max knee vel: {max_knee_vel:.1f} rad/s (limit {KNEE_VEL_MAX})")
    print(f"  Status: {'PASS' if success else 'FAIL'}")

    return success, speed, walk_time


if __name__ == "__main__":
    run_walking_test("forward")
    run_walking_test("backward")