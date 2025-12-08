import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import imageio


# ================================================================
#                MUJOCO 2.3 NAME HELPERS
# ================================================================
def mj_name(model, adr):
    end = model.names.find(b"\x00", adr)
    return model.names[adr:end].decode()

def joint_name(model, j_id):
    adr = model.name_jntadr[j_id]
    return mj_name(model, adr)

def joint_id(model, name):
    for j in range(model.njnt):
        if joint_name(model, j) == name:
            return j
    raise ValueError(f"Joint '{name}' not found")

def actuator_joint_id(model, actuator_id):
    return model.actuator_trnid[actuator_id][0]


# ================================================================
#              LIMIT CONSTANTS
# ================================================================
HIP_ANGLE_MIN, HIP_ANGLE_MAX = -2.094, 0.524
KNEE_ANGLE_MIN, KNEE_ANGLE_MAX = 0.0, 2.793

HIP_VEL_MAX  = 30.0
KNEE_VEL_MAX = 15.0

HIP_TAU_MAX  = 30.0
KNEE_TAU_MAX = 60.0

HIP_JOINTS  = ["q1", "q3"]
KNEE_JOINTS = ["q2", "q4"]
LEG_JOINTS  = HIP_JOINTS + KNEE_JOINTS
ROOT_JOINTS = ["root_x", "root_z", "root_theta"]


# ================================================================
#              TORQUE SATURATION
# ================================================================
def saturate_torques(model, tau_cmd):
    tau = tau_cmd.copy()
    for i in range(model.nu):
        jid = actuator_joint_id(model, i)
        jn = joint_name(model, jid)
        limit = HIP_TAU_MAX if jn in HIP_JOINTS else KNEE_TAU_MAX
        tau[i] = np.clip(tau[i], -limit, limit)
    return tau


# ================================================================
#              LIMIT CHECKER
# ================================================================
def check_limits(model, data, active_joint):
    """Only check limits for the active joint"""
    jid = joint_id(model, active_joint)
    q = data.qpos[jid]
    dq = data.qvel[jid]
    
    if active_joint in HIP_JOINTS:
        if not (HIP_ANGLE_MIN <= q <= HIP_ANGLE_MAX):
            return f"Hip ANGLE violation: {active_joint} = {q:.3f} rad ({np.rad2deg(q):.1f}°)"
        if abs(dq) > HIP_VEL_MAX:
            return f"Hip VELOCITY violation: {active_joint} = {dq:.3f} rad/s"
    
    if active_joint in KNEE_JOINTS:
        if not (KNEE_ANGLE_MIN <= q <= KNEE_ANGLE_MAX):
            return f"Knee ANGLE violation: {active_joint} = {q:.3f} rad ({np.rad2deg(q):.1f}°)"
        if abs(dq) > KNEE_VEL_MAX:
            return f"Knee VELOCITY violation: {active_joint} = {dq:.3f} rad/s"
    
    return None


# ================================================================
#       RUN ONE SCENARIO (HARD FREEZE NON-ACTIVE JOINTS)
# ================================================================
def run_scenario(model, window, cam, opt, scene, ctx,
                 scenario_name, active_joint, torque_script,
                 max_time=3.0, fps=60):

    print(f"\n=== RUNNING SCENARIO: {scenario_name} ===")
    print("Active joint =", active_joint)

    data = mj.MjData(model)

    # Gravity OFF
    model.opt.gravity[:] = 0.0

    # Standard starting pose
    default_angles = {
        "q1": 0.0,
        "q3": 0.0,
        "q2": 1.396,     # ~80°
        "q4": 1.396,
    }
    
    # Set initial joint positions
    for jn, val in default_angles.items():
        data.qpos[joint_id(model, jn)] = val

    # Zero velocities
    data.qvel[:] = 0.0
    
    # Forward kinematics
    mj.mj_forward(model, data)

    frames = []

    # Get ID of active joint & actuator index
    active_jid = joint_id(model, active_joint)
    active_act = None
    for ai in range(model.nu):
        if actuator_joint_id(model, ai) == active_jid:
            active_act = ai
            break

    # ============================================================
    # SIMULATION LOOP
    # ============================================================
    while (not glfw.window_should_close(window)) and data.time < max_time:

        glfw.poll_events()

        # Apply torque only to active joint
        tau_cmd = np.zeros(model.nu)
        tau_cmd[active_act] = torque_script(data.time)
        
        # For non-active joints, apply strong position-holding torques
        for jn in LEG_JOINTS:
            if jn != active_joint:
                jid = joint_id(model, jn)
                # Find actuator for this joint
                for ai in range(model.nu):
                    if actuator_joint_id(model, ai) == jid:
                        # PD controller to hold position
                        q_error = default_angles[jn] - data.qpos[jid]
                        v_error = 0.0 - data.qvel[jid]
                        tau_cmd[ai] = 1000.0 * q_error + 100.0 * v_error
                        break
        
        # Apply saturated torques
        data.ctrl[:] = saturate_torques(model, tau_cmd)

        # Step dynamics
        mj.mj_step(model, data)

        # Force root joints to zero (freeze torso in space)
        for jn in ROOT_JOINTS:
            jid = joint_id(model, jn)
            data.qpos[jid] = 0.0
            data.qvel[jid] = 0.0

        # Check for violations on active joint only
        err = check_limits(model, data, active_joint)

        # Render
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, fb_w, fb_h)

        mj.mjv_updateScene(model, data, opt, None, cam,
                           mj.mjtCatBit.mjCAT_ALL, scene)

        mj.mjr_render(viewport, scene, ctx)

        # Get current joint state for display
        q_curr = data.qpos[joint_id(model, active_joint)]
        dq_curr = data.qvel[joint_id(model, active_joint)]
        tau_curr = data.ctrl[active_act]

        # Overlay text with joint state
        status_line = f"q={q_curr:.3f} ({np.rad2deg(q_curr):.1f}°)  dq={dq_curr:.2f} rad/s  τ={tau_curr:.1f} Nm"
        
        mj.mjr_overlay(
            mj.mjtFontScale.mjFONTSCALE_150,
            mj.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            f"{scenario_name}   t={data.time:.2f}s",
            status_line + ("  " + err if err else ""),
            ctx
        )

        # Save frame
        rgb = np.zeros((fb_h, fb_w, 3), dtype=np.uint8)
        mj.mjr_readPixels(rgb, None, viewport, ctx)
        rgb = np.flipud(rgb)
        frames.append(rgb.copy())

        glfw.swap_buffers(window)

        if err:
            print("TERMINATED:", err)
            break

    # Save video
    os.makedirs("videos", exist_ok=True)
    path = f"videos/{scenario_name}.mp4"
    imageio.mimwrite(path, frames, fps=fps)
    print(f"Saved: {path} ({len(frames)} frames)")


# ================================================================
#                     SCENARIOS
# ================================================================
def build_scenarios(model, window, cam, opt, scene, ctx):
    """
    Test each constraint violation scenario
    """
    scenarios = [
        # Hip angle violations (start at 0°)
        ("q1_angle_lower_violation", "q1", lambda t: -20.0),  # constant torque to go negative
        ("q1_angle_upper_violation", "q1", lambda t: +20.0),  # constant torque to go positive
        ("q3_angle_lower_violation", "q3", lambda t: -20.0),
        ("q3_angle_upper_violation", "q3", lambda t: +20.0),
        
        # Knee angle violations (start at 80°)
        ("q2_angle_lower_violation", "q2", lambda t: -40.0),  # torque to decrease angle
        ("q2_angle_upper_violation", "q2", lambda t: +40.0),  # torque to increase angle
        ("q4_angle_lower_violation", "q4", lambda t: -40.0),
        ("q4_angle_upper_violation", "q4", lambda t: +40.0),
        
        # Hip velocity violations
        ("q1_velocity_violation", "q1", lambda t: 30.0 if t < 1.0 else 0.0),  # ramp up velocity
        ("q3_velocity_violation", "q3", lambda t: 30.0 if t < 1.0 else 0.0),
        
        # Knee velocity violations
        ("q2_velocity_violation", "q2", lambda t: 60.0 if t < 0.5 else 0.0),  # ramp up velocity
        ("q4_velocity_violation", "q4", lambda t: 60.0 if t < 0.5 else 0.0),
    ]

    for name, active_joint, torque_fn in scenarios:
        run_scenario(model, window, cam, opt, scene, ctx,
                     name, active_joint, torque_fn, max_time=5.0)


# ================================================================
#                           MAIN
# ================================================================
def main():
    model = mj.MjModel.from_xml_path("biped_robot.xml")

    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW")

    window = glfw.create_window(1280, 720, "Biped Joint Limit Violations", None, None)
    glfw.make_context_current(window)

    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)

    # Camera settings - side view for 2D biped
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array([0.0, 0.0, 0.5])
    cam.distance = 2.5
    cam.azimuth = 90
    cam.elevation = -15

    scene = mj.MjvScene(model, maxgeom=20000)
    ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)

    build_scenarios(model, window, cam, opt, scene, ctx)

    glfw.terminate()
    print("\n=== ALL SCENARIOS COMPLETED ===")


if __name__ == "__main__":
    main()