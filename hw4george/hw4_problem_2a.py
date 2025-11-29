from pathlib import Path
import numpy as np
import imageio
import mujoco as mj
import matplotlib.pyplot as plt
import cvxpy as cp

# ============================= CONFIG =============================
XML_NAME = "biped2d_hw4_CORRECTED.xml"
DUR = 2.0
RECORD_MP4 = True
MP4_NAME = "hw4_q2a_biped.mp4"
FPS = 60

# Desired state
X_DES = 0.0
Z_DES = 0.5
THETA_DES = 0.0

MU = 0.7

# Gains - VERY STRONG for accurate tracking
KPX, KDX = 200.0, 40.0
KPZ, KDZ = 3500.0, 700.0  # Even stronger!
KPT, KDT = 2500.0, 300.0

# Small posture PD
KP_POST = 600.0
KD_POST = 60.0
q_des = np.zeros(4)

# Initial angles from HW4
q1_hw = -np.pi / 3
q3_hw = -np.pi / 6
q2_hw = np.pi / 2
q4_hw = np.pi / 2
q_hw = np.array([q1_hw, q2_hw, q3_hw, q4_hw])
q_mj_init = q_hw


def load_model():
    p = Path(__file__).with_name(XML_NAME)
    model = mj.MjModel.from_xml_path(str(p))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return model, data


def joint_ids(model):
    return [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in ("q1", "q2", "q3", "q4")]


def trunk_pitch(model, data):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "trunk")
    R = data.xmat[bid].reshape(3, 3)
    return np.arctan2(R[0, 2], R[2, 2])


def set_initial_state(model, data):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.6  # 60cm - reasonable fall height
    data.qpos[3] = 1.0

    jids = joint_ids(model)
    for i, jid in enumerate(jids):
        data.qpos[model.jnt_qposadr[jid]] = q_mj_init[i]

    mj.mj_forward(model, data)

    # Robot will fall from initial height specified in qpos[2]!

    # CRITICAL: make initial COM x exactly 0.0
    current_com_x = data.subtree_com[1][0]
    data.qpos[0] -= current_com_x
    mj.mj_forward(model, data)

    print(f"Initial COM: x={data.subtree_com[1][0]:.6f}, y={data.subtree_com[1][2]:.3f}")


def both_feet_loaded(model, data):
    # Check if toe geoms are in contact with floor
    l_toe_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "l_toe")
    r_toe_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "r_toe")
    floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")

    left_contact = False
    right_contact = False

    for i in range(data.ncon):
        c = data.contact[i]
        # Check if contact involves floor and a toe
        if (c.geom1 == floor_id and c.geom2 == l_toe_id) or (c.geom2 == floor_id and c.geom1 == l_toe_id):
            left_contact = True
        if (c.geom1 == floor_id and c.geom2 == r_toe_id) or (c.geom2 == floor_id and c.geom1 == r_toe_id):
            right_contact = True

    # Debug occasionally
    if np.random.rand() < 0.001:
        print(f"  [Debug] Contacts: left={left_contact}, right={right_contact}, total={data.ncon}")

    return left_contact and right_contact


def solve_qp(model, data, com, vcom, theta, omega):
    ax_des = -KPX * (com[0] - X_DES) - KDX * vcom[0]
    az_des = -KPZ * (com[2] - Z_DES) - KDZ * vcom[2]
    ath_des = -KPT * (theta - THETA_DES) - KDT * omega

    ax_des = np.clip(ax_des, -20, 20)
    az_des = np.clip(az_des, -60, 60)  # Allow stronger vertical
    ath_des = np.clip(ath_des, -150, 150)

    m = np.sum(model.body_mass)
    g = -model.opt.gravity[2]
    I = m * (Z_DES ** 2) * 40

    ltoe = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "l_toe")
    rtoe = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "r_toe")
    pL = data.geom_xpos[ltoe]
    pR = data.geom_xpos[rtoe]

    A = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [pL[2] - com[2], -(pL[0] - com[0]), pR[2] - com[2], -(pR[0] - com[0])]
    ])
    b = np.array([m * ax_des, m * (az_des + g), I * ath_des])

    jacp_L = np.zeros((3, model.nv))
    jacp_R = np.zeros((3, model.nv))
    left_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "left_shank")
    right_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "right_shank")
    mj.mj_jacBody(model, data, jacp_L, None, left_bid)
    mj.mj_jacBody(model, data, jacp_R, None, right_bid)

    A_tau = np.zeros((4, 4))
    jids = joint_ids(model)
    for i, jid in enumerate(jids):
        da = model.jnt_dofadr[jid]
        A_tau[i, 0] = jacp_L[0, da]
        A_tau[i, 1] = jacp_L[2, da]
        A_tau[i, 2] = jacp_R[0, da]
        A_tau[i, 3] = jacp_R[2, da]

    f = cp.Variable(4)
    cost = cp.sum_squares(A @ f - b) + 1e-3 * cp.sum_squares(f) + 15 * cp.square(f[1] - f[3])

    constraints = [
        f[1] >= 10, f[1] <= 250,
        f[3] >= 10, f[3] <= 250,
        f[0] >= -MU * f[1], f[0] <= MU * f[1],
        f[2] >= -MU * f[3], f[2] <= MU * f[3],
    ]
    for i in range(4):
        constraints += [A_tau[i, :] @ f <= 150000, A_tau[i, :] @ f >= -150000]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)

    if prob.status in ["optimal", "optimal_inaccurate"]:
        return A_tau @ f.value
    return np.zeros(4)


# ============================= MAIN =============================
model, data = load_model()
set_initial_state(model, data)

if RECORD_MP4:
    renderer = mj.Renderer(model, height=720, width=1280)
    cam = mj.MjvCamera()
    cam.distance = 1.3
    cam.elevation = -15
    cam.azimuth = 90
    cam.lookat = np.array([0.0, 0.0, 0.5])
    writer = imageio.get_writer(MP4_NAME, fps=FPS)
    steps_per_frame = max(1, int(1 / FPS / model.opt.timestep))

log_t, log_x, log_z, log_theta, log_tau = [], [], [], [], []

prev_com = None
prev_theta = None
controller_active_count = 0
total_steps = 0

print(f"\nQP Controller with KPZ={KPZ}, KDZ={KDZ}\n")
print("Starting simulation - watching for controller activation...\n")

while data.time < DUR:
    total_steps += 1

    # QP balancing
    com = data.subtree_com[1].copy()
    vcom = (com - prev_com) / model.opt.timestep if prev_com is not None else np.zeros(3)
    theta = trunk_pitch(model, data)
    omega = (theta - prev_theta) / model.opt.timestep if prev_theta is not None else 0.0

    feet_loaded = both_feet_loaded(model, data)

    if feet_loaded:
        # ONLY NOW apply gravity compensation and posture control
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mj.mj_inverse(model, data)
        tau_grav = data.qfrc_inverse[6:10].copy()

        # Small posture PD + gravity compensation
        q_err = q_des - data.qpos[7:11]
        qd = data.qvel[6:10]
        tau = KP_POST * q_err - KD_POST * qd + tau_grav

        controller_active_count += 1
        if controller_active_count == 1:
            print(f"⚠️ Controller activated at t={data.time:.3f}s, z={com[2]:.3f}m")
        tau += solve_qp(model, data, com, vcom, theta, omega)
    else:
        # Free fall - no control!
        tau = np.zeros(4)

    # Debug output every 0.1s
    if total_steps % 400 == 0:
        print(f"t={data.time:.2f}s: z={com[2]:.3f}m, feet_loaded={feet_loaded}, contacts={data.ncon}")

    data.qfrc_applied[6:10] = tau

    mj.mj_step(model, data)

    log_t.append(data.time)
    log_x.append(com[0])
    log_z.append(com[2])
    log_theta.append(theta)
    log_tau.append(tau.copy())

    if RECORD_MP4 and (int(data.time / model.opt.timestep) % steps_per_frame == 0):
        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())

    prev_com = com.copy()
    prev_theta = theta

if RECORD_MP4:
    writer.close()

# Plots
log_tau = np.array(log_tau)
plt.figure(figsize=(10, 10))

plt.subplot(4, 1, 1)
plt.plot(log_t, log_x, 'b-', linewidth=2)
plt.axhline(X_DES, color='r', linestyle='--', label='Desired')
plt.ylabel("x(t) [m]")
plt.title("Trunk Horizontal Position")
plt.legend()
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(log_t, log_z, 'g-', linewidth=2)
plt.axhline(Z_DES, color='r', linestyle='--', label='Desired')
plt.ylabel("y(t) [m]")
plt.title("Trunk Vertical Position (Height)")
plt.legend()
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(log_t, np.rad2deg(log_theta), 'orange', linewidth=2)
plt.axhline(np.rad2deg(THETA_DES), color='r', linestyle='--', label='Desired')
plt.ylabel("θ(t) [deg]")
plt.title("Trunk Pitch Angle")
plt.legend()
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(log_t, log_tau[:, 0], linewidth=2, label="q1 (left hip)")
plt.plot(log_t, log_tau[:, 1], linewidth=2, label="q2 (left knee)")
plt.plot(log_t, log_tau[:, 2], linewidth=2, label="q3 (right hip)")
plt.plot(log_t, log_tau[:, 3], linewidth=2, label="q4 (right knee)")
plt.legend()
plt.xlabel("time [s]")
plt.ylabel("torque [Nm]")
plt.title("Joint Torques")
plt.grid()

plt.tight_layout()
plt.savefig('hw4_2a_plots.png', dpi=150)

print(f"\n{'=' * 60}")
print(f"QP Controller Results")
print(f"{'=' * 60}")
print(f"Final x: {log_x[-1]:.4f} m → error: {abs(log_x[-1]):.4f} m")
print(f"Final y: {log_z[-1]:.4f} m → error: {abs(log_z[-1] - Z_DES):.4f} m")
print(f"Final θ: {np.rad2deg(log_theta[-1]):.2f}°")
print(f"Peak torque: {np.max(np.abs(log_tau)):.1f} Nm")
print(f"{'=' * 60}")

plt.show()