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
MP4_NAME = "hw4_2b_mpc.mp4"
FPS = 60

# MPC parameters
N_HORIZON = 10
DT_MPC = 0.04

# Desired state
X_DES = 0.0
Z_DES = 0.5
THETA_DES = 0.0
MU = 0.7

# Small posture PD
KP_POST = 600.0
KD_POST = 60.0
q_des = np.zeros(4)

# Initial angles
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

    # Robot falls from initial height qpos[2]!

    current_com_x = data.subtree_com[1][0]
    data.qpos[0] -= current_com_x
    mj.mj_forward(model, data)

    print(f"Initial: x={data.subtree_com[1][0]:.6f}, z={data.subtree_com[1][2]:.3f}")


def both_feet_loaded(model, data):
    # Check if toe geoms are in contact with floor
    l_toe_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "l_toe")
    r_toe_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "r_toe")
    floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")

    left_contact = False
    right_contact = False

    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == floor_id and c.geom2 == l_toe_id) or (c.geom2 == floor_id and c.geom1 == l_toe_id):
            left_contact = True
        if (c.geom1 == floor_id and c.geom2 == r_toe_id) or (c.geom2 == floor_id and c.geom1 == r_toe_id):
            right_contact = True

    return left_contact and right_contact


def solve_linearized_mpc(model, data, com, vcom, theta, omega):
    """
    TRUE MPC: Optimize force trajectory over N timesteps using linearized dynamics.
    Key: optimize u[0], u[1], ..., u[N-1] subject to x[k+1] = f(x[k], u[k])
    """
    m = np.sum(model.body_mass)
    g = -model.opt.gravity[2]

    # Get foot positions
    ltoe = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "l_toe")
    rtoe = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "r_toe")
    pL = data.geom_xpos[ltoe].copy()
    pR = data.geom_xpos[rtoe].copy()

    # Current state [x, z, theta, vx, vz, omega]
    x0 = np.array([com[0], com[2], theta, vcom[0], vcom[2], omega])

    # MPC decision variables
    # x[k] for k=0..N (states)
    # u[k] for k=0..N-1 (controls = forces)
    x_mpc = cp.Variable((6, N_HORIZON + 1))
    u_mpc = cp.Variable((4, N_HORIZON))  # [FxL, FzL, FxR, FzR]

    # Cost function
    cost = 0
    x_ref = np.array([X_DES, Z_DES, THETA_DES, 0, 0, 0])

    # Weights (MATCH 2a gains for fair comparison)
    W_state = np.array([200, 3500, 2500, 40, 700, 300])  # [x, z, theta, vx, vz, omega]

    for k in range(N_HORIZON):
        # State tracking cost
        for i in range(6):
            cost += W_state[i] * cp.square(x_mpc[i, k] - x_ref[i])

        # Control cost
        cost += 0.01 * cp.sum_squares(u_mpc[:, k])
        cost += 20 * cp.square(u_mpc[1, k] - u_mpc[3, k])  # Symmetric forces

    # Terminal cost (extra weight)
    for i in range(6):
        cost += 2 * W_state[i] * cp.square(x_mpc[i, N_HORIZON] - x_ref[i])

    # Constraints
    constraints = []

    # Initial state
    constraints.append(x_mpc[:, 0] == x0)

    # Dynamics: x[k+1] = x[k] + dx/dt * dt
    # dx/dt = [vx, vz, omega, ax, az, alpha]
    # where ax = (FxL + FxR)/m
    #       az = (FzL + FzR)/m - g
    #       alpha = moment / I

    I_approx = m * (com[2] ** 2) * 40

    # Linearize dynamics around current state x0
    # dx/dt = [vx, vz, omega, ax, az, alpha]
    # ax = (FxL + FxR)/m
    # az = (FzL + FzR)/m - g
    # alpha = M/I where M = moment computed at x0

    # Moment arms at current state (constants!)
    r_Lz = pL[2] - x0[1]  # Vertical moment arm for left foot
    r_Lx = pL[0] - x0[0]  # Horizontal moment arm for left foot
    r_Rz = pR[2] - x0[1]  # Vertical moment arm for right foot
    r_Rx = pR[0] - x0[0]  # Horizontal moment arm for right foot

    # Build linear system: dx/dt = A*x + B*u + c
    # States: [x, z, theta, vx, vz, omega]
    # Controls: [FxL, FzL, FxR, FzR]

    A_dyn = np.array([
        [0, 0, 0, 1, 0, 0],  # dx/dt = vx
        [0, 0, 0, 0, 1, 0],  # dz/dt = vz
        [0, 0, 0, 0, 0, 1],  # dtheta/dt = omega
        [0, 0, 0, 0, 0, 0],  # dvx/dt = ax (from forces)
        [0, 0, 0, 0, 0, 0],  # dvz/dt = az (from forces)
        [0, 0, 0, 0, 0, 0],  # domega/dt = alpha (from moment)
    ])

    B_dyn = np.array([
        [0, 0, 0, 0],  # x
        [0, 0, 0, 0],  # z
        [0, 0, 0, 0],  # theta
        [1 / m, 0, 1 / m, 0],  # vx: affected by FxL and FxR
        [0, 1 / m, 0, 1 / m],  # vz: affected by FzL and FzR
        [r_Lz / I_approx, -r_Lx / I_approx, r_Rz / I_approx, -r_Rx / I_approx]  # omega: moment
    ])

    c_dyn = np.array([0, 0, 0, 0, -g, 0])  # Gravity constant

    # Forward Euler integration: x[k+1] = x[k] + dt*(A*x[k] + B*u[k] + c)
    for k in range(N_HORIZON):
        x_next = x_mpc[:, k] + DT_MPC * (A_dyn @ x_mpc[:, k] + B_dyn @ u_mpc[:, k] + c_dyn)
        constraints.append(x_mpc[:, k + 1] == x_next)

        # Force constraints
        constraints.append(u_mpc[1, k] >= 10)  # FzL >= 10
        constraints.append(u_mpc[1, k] <= 250)  # FzL <= 250
        constraints.append(u_mpc[3, k] >= 10)  # FzR >= 10
        constraints.append(u_mpc[3, k] <= 250)  # FzR <= 250

        # Friction cone
        constraints.append(u_mpc[0, k] >= -MU * u_mpc[1, k])
        constraints.append(u_mpc[0, k] <= MU * u_mpc[1, k])
        constraints.append(u_mpc[2, k] >= -MU * u_mpc[3, k])
        constraints.append(u_mpc[2, k] <= MU * u_mpc[3, k])

    # Solve
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, warm_start=True, max_iter=10000, eps_abs=1e-5, eps_rel=1e-5, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        if np.random.rand() < 0.1:  # Print occasionally
            print(f"  MPC FAILED: {prob.status} at t={data.time:.2f}s - using QP fallback")

        # Fallback to QP controller
        ax_des = -200 * (x0[0] - X_DES) - 40 * x0[3]
        az_des = -3500 * (x0[1] - Z_DES) - 700 * x0[4]
        ath_des = -2500 * (x0[2] - THETA_DES) - 300 * x0[5]

        ax_des = np.clip(ax_des, -20, 20)
        az_des = np.clip(az_des, -60, 60)
        ath_des = np.clip(ath_des, -150, 150)

        # Solve single-step QP
        A_eq = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [pL[2] - x0[1], -(pL[0] - x0[0]), pR[2] - x0[1], -(pR[0] - x0[0])]
        ])
        b_eq = np.array([m * ax_des, m * (az_des + g), I_approx * ath_des])

        f_qp = cp.Variable(4)
        cost_qp = cp.sum_squares(A_eq @ f_qp - b_eq) + 1e-3 * cp.sum_squares(f_qp)
        constraints_qp = [
            f_qp[1] >= 10, f_qp[1] <= 250,
            f_qp[3] >= 10, f_qp[3] <= 250,
            f_qp[0] >= -MU * f_qp[1], f_qp[0] <= MU * f_qp[1],
            f_qp[2] >= -MU * f_qp[3], f_qp[2] <= MU * f_qp[3],
        ]
        prob_qp = cp.Problem(cp.Minimize(cost_qp), constraints_qp)
        prob_qp.solve(solver=cp.OSQP, warm_start=True)

        if prob_qp.status in ["optimal", "optimal_inaccurate"]:
            optimal_forces = f_qp.value
        else:
            return np.array([0, m * g / 2, 0, m * g / 2])
    else:
        # MPC succeeded - use its solution
        optimal_forces = u_mpc.value[:, 0]

    # Debug output occasionally
    if np.random.rand() < 0.02:
        predicted_z = x_mpc.value[1, -1]
        print(f"  MPC at t={data.time:.2f}s: z={x0[1]:.3f} → predicted z[N]={predicted_z:.3f} (want {Z_DES})")
        print(f"       Forces: FzL={optimal_forces[1]:.1f}, FzR={optimal_forces[3]:.1f}")

    # Convert forces to torques
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

    return A_tau @ optimal_forces


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
mpc_counter = 0
steps_per_mpc = int(DT_MPC / model.opt.timestep)

print(f"\nLinearized MPC: N={N_HORIZON}, dt={DT_MPC}s, updates every {steps_per_mpc} steps\n")

step = 0
while data.time < DUR:
    step += 1

    com = data.subtree_com[1].copy()
    vcom = (com - prev_com) / model.opt.timestep if prev_com is not None else np.zeros(3)
    theta = trunk_pitch(model, data)
    omega = (theta - prev_theta) / model.opt.timestep if prev_theta is not None else 0.0

    if both_feet_loaded(model, data):
        # ONLY apply control after landing
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mj.mj_inverse(model, data)
        tau_grav = data.qfrc_inverse[6:10].copy()

        q_err = q_des - data.qpos[7:11]
        qd = data.qvel[6:10]
        tau = KP_POST * q_err - KD_POST * qd + tau_grav

        if (mpc_counter % steps_per_mpc == 0):
            tau_mpc = solve_linearized_mpc(model, data, com, vcom, theta, omega)
            tau += tau_mpc
            mpc_counter += 1
        elif both_feet_loaded(model, data):
            mpc_counter += 1
    else:
        # Free fall - no control
        tau = np.zeros(4)

    data.qfrc_applied[6:10] = tau
    mj.mj_step(model, data)

    log_t.append(data.time)
    log_x.append(com[0])
    log_z.append(com[2])
    log_theta.append(theta)
    log_tau.append(tau.copy())

    if RECORD_MP4 and (step % steps_per_frame == 0):
        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())

    prev_com = com.copy()
    prev_theta = theta

if RECORD_MP4:
    writer.close()

# Plots
log_tau = np.array(log_tau)
fig = plt.figure(figsize=(10, 10))

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
plt.savefig('hw4_2b_plots.png', dpi=150)

print(f"\n{'=' * 60}")
print(f"Linearized MPC Results (N={N_HORIZON}, dt={DT_MPC}s)")
print(f"{'=' * 60}")
print(f"Final x: {log_x[-1]:.4f} m → error: {abs(log_x[-1]):.4f} m")
print(f"Final y: {log_z[-1]:.4f} m → error: {abs(log_z[-1] - Z_DES):.4f} m")
print(f"Final θ: {np.rad2deg(log_theta[-1]):.2f}°")
print(f"Peak torque: {np.max(np.abs(log_tau)):.1f} Nm")
print(f"{'=' * 60}")

plt.show()