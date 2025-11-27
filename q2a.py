import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
from cvxopt import matrix, solvers
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # disable runtime errors from cvxopt
os.environ["OMP_NUM_THREADS"] = "1"           # disable runtime errors from cvxopt
solvers.options['show_progress'] = False      # disable printing from cvxopt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.utils import *


def plot(t, data):
    plot_dir = "hw4/plots"
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(11, 9))
    
    plt.subplot(4,2,1)
    plt.plot(t, data[:, 0], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('x(t) [m]')
    plt.grid()

    plt.subplot(4,2,2)
    plt.plot(t, data[:, 1], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('y(t) [m]')    
    plt.grid()

    plt.subplot(4,2,3)
    plt.plot(t,  data[:, 2], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('θ(t) [rad]')
    plt.grid()
    
    plt.subplot(4,2,4)
    plt.plot(t,  data[:, 3], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('τ1(t) [Nm]')
    plt.grid()
    
    plt.subplot(4,2,5)
    plt.plot(t,  data[:, 4], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('τ2(t) [Nm]')
    plt.grid()
    
    plt.subplot(4,2,6)
    plt.plot(t,  data[:, 5], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('τ3(t) [Nm]')
    plt.grid()
    
    plt.subplot(4,2,7)
    plt.plot(t,  data[:, 6], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('τ4(t) [Nm]')
    plt.grid()
    
    plt.suptitle("x(t), θ(t), τ(t) for balancing biped under qp control")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/q2a.pdf")
    plt.close()
    

def qp(x, m, d):
    
    if not get_feet_contact(m, d): return np.zeros(m.nu)
    
    # cvxopt solves:
    # xi* = min_x (1/2) xTPx + qTx
    # s.t. Ax = b    (equality constraints)
    # s.t. Gx ≤ h    (inequality constraints)
    
    # The problem statement is:
    # F* = min_F (AF - b).TQ(AF - b) + (αF.T)F
    # s.t. 10 ≤ |Fy| ≤ 250
    # s.t. |Fx/Fy| ≤ μ
    
    # Expand the optimization:
    # F* = min_F (AF - b).TQ(AF - b) + α(F.T)F
    #    = min_F F.T(A.T)Q(AF) - F.T(A.T)Qb - b.TQ(AF) + b.TQb + α(F.T)F
    #    = min_F F.T(A.T)Q(AF) - F.T(A.T)Qb - b.TQ(AF) + α(F.T)F
    #    = min_F F.T( A.TQA + αI )F - (2b.TQA)F
    # b.TQb can be discarded because constant terms in the QP formulation do not affect the solution
    
    # Rewrite the contact force inequality constraint:
    # s.t. 10 ≤ |Fy| ≤ 250
    # s.t.  Fy ≤ 250    
    # s.t. -Fy ≤ -10
    # We disregard the following bounds:
    # s.t. Fy ≥ -250 -> -Fy ≤ 250 (the lower bound on Fy is already handled by -Fy ≤ -10. Moreover, Fy cannot be negative)
    # s.t. Fy ≤ -10               (Fy cannot be negative)

    # In matrix form...
    # [0  1  0  0] F ≤ [250]
    # [0  0  0  1]     [250]
    # [0 -1  0  0]     [-10]
    # [0  0  0 -1]     [-10]
    # row 1 gives:      0F1x + F1y + 0F2x + 0F2y ≤ 250
    # row 2 gives:      0F1x + 0F1y + 0F2x + F2y ≤ 250
    # row 3 gives:      0F1x - F1y + 0F2x + 0F2y ≤ -10 (or equivalently:  F1y ≥ 10)
    # row 4 gives:      0F1x + 0F1y + 0F2x + F2y ≤ -10 (or equivalently:  F2y ≥ 10)

    # Rewrite the friction inequality constraint:
    # s.t. |Fx/Fy| ≤ μ
    # --------------------
    # s.t.  Fx/Fy ≤ μ
    # s.t.  Fx ≤ μFy
    # s.t.  Fx - μFy ≤ 0
    # --------------------
    # s.t. -Fx/Fy ≤ μ  
    # s.t. -Fx ≤ μFy
    # s.t. -Fx - μFy ≤ 0
    # Note that we must rearrange these inequalities because decision variables (Fy) are NOT allowed on the h vector

    # In matrix form...
    # [1  -μ  0   0] F ≤ [0]
    # [0   0  1  -μ]     [0]
    # [-1 -μ  0   0]     [0]
    # [0   0 -1  -μ]     [0]
    # row 1 gives:      F1x - μF1y + 0F2x + 0F2y  ≤ 0   (or equivalently:  F1x/F1y ≤ μ)
    # row 2 gives:      0F1x + 0F1y + F2x - μF2y  ≤ 0   (or equivalently:  F2x/F2y ≤ μ)
    # row 3 gives:      -F1x - μF1y + 0F2x + 0F2y ≤ 0   (or equivalently:  -F1x/F1y ≤ μ,  F1x/F1y ≥ -μ)
    # row 4 gives:      0F1x + 0F1y - F2x - μF2y  ≤ 0   (or equivalently:  -F2x/F2y ≤ μ,  F2x/F2y ≥ -μ)
    
    # Thus:
    # P =  2(A.TQA + αI)
    # q = -2A.TQb
    # G = [0  1   0   0]
    #     [0  0   0   1]
    #     [0 -1   0   0]
    #     [0  0   0  -1]
    #     [1  -μ  0   0]
    #     [0   0  1  -μ]
    #     [-1 -μ  0   0]
    #     [0   0 -1  -μ]
    
    # h = [250]
    #     [250]
    #     [-10]
    #     [-10]
    #     [ 0 ]
    #     [ 0 ]
    #     [ 0 ]
    #     [ 0 ]
        
    # Generically
    # Let n: # decision variables, i: # of equality constraints, j: # of inequality constraints
    # P ∈ Rnxn
    # q ∈ Rnx1    
    # A ∈ Rixn
    # b ∈ Rix1
    # G ∈ Rjxn
    # h ∈ Rjx1
            
    # M = get_body_mass(m, "torso")
    M = get_body_mass(m, "torso") + get_body_mass(m, "l_thigh") + get_body_mass(m, "l_calf") + get_body_mass(m, "r_thigh") + get_body_mass(m, "r_calf") 
    Izz = get_body_inertia(m, "torso")[1]
    # Izz_torso = (1/12) * M * (a^2 + b^2) = (1/12) * 8 * ( 0.25^2 + 0.15^2 ) = 0.05666666667
    # Izz_leg  = (1/12) * mi * l^2 = (1/12) * 0.25 * 0.22^2 = 0.001008333333
    g = np.abs(get_gravity(m))
    
    # Define cost matrices
    alpha = 0.001
    Q = np.array([
        [1,   0,  0.1],    
        [0,   1,    0],    
        [0.1, 0,    5],    
    ])

    q_pos = x[:m.nq]
    q_vel = x[m.nq:]
    
    xc = q_pos[0]
    yc = q_pos[2]
    theta_c_quat = q_pos[3:7]
    theta_c = R.from_quat(theta_c_quat, scalar_first=True).as_euler('zyx')[1]
    
    xc_dot = q_vel[0]
    yc_dot = q_vel[2]
    theta_c_dot = q_vel[4]
    
    xc_des = 0
    yc_des = 0.5
    theta_c_des = 0 
    xc_dot_des = 0
    yc_dot_des = 0
    theta_c_dot_des = 0
    
    Kp_x = 5
    Kd_x = 10
    Kp_y = 5
    Kd_y = 10
    Kp_theta = 0.1
    Kd_theta = 0.1

    x_ddot_des = Kp_x * (xc_des - xc) + Kd_x * (xc_dot_des - xc_dot)
    y_ddot_des = Kp_y * (yc_des - yc) + Kd_y * (yc_dot_des - yc_dot)
    theta_ddot_des = Kp_theta * (theta_c_des - theta_c) + Kd_theta * (theta_c_dot_des - theta_c_dot)
    
    PF1, PF2 = get_feet_xpos(m, d)[:, :]
    PF1x, PF1y = PF1[[0, 2]]
    PF2x, PF2y = PF2[[0, 2]]
    
    # Uses the definition F = [F1x F1y F2x F2y] 
    A = np.array([
        [1,       0,       1,       0      ],
        [0,       1,       0,       1      ],
        [yc-PF1y, PF1x-xc, yc-PF2y, PF2x-xc]
    ])
    
    b = np.array([
        [M * x_ddot_des],
        [M * (y_ddot_des + g)],
        [Izz * theta_ddot_des],
        
    ])
    
    n = A.shape[1]
    
    Fy_max = 250
    Fy_min = 10
    mu = 0.7

    P = 2 * (A.T @ Q @ A + alpha * np.eye(n))
    q = -2 * A.T @ Q @ b
    G = np.array([
        [0,    1,   0,   0],
        [0,    0,   0,   1],
        [0,   -1,   0,   0],
        [0,    0,   0,  -1],
        [1,  -mu,   0,   0],
        [0,    0,   1, -mu],
        [-1, -mu,   0,   0],
        [0,    0,  -1, -mu]
    ])
    h = np.array([
        [Fy_max],
        [Fy_max],
        [-Fy_min],
        [-Fy_min],
        [0],
        [0],
        [0],
        [0]
    ])
    
    # Convert numpy matrices to cvxopt matrices
    P_cvx = matrix(P.astype(float))
    q_cvx = matrix(q.astype(float))
    G_cvx = matrix(G.astype(float))
    h_cvx = matrix(h.astype(float))
    
    # Solve
    sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    F_GRF_star = np.array([sol['x']])[0]
    
    F_feet = -F_GRF_star
    
    F_foot_l = np.vstack([ F_feet[0], 0, F_feet[1] ])
    F_foot_r = np.vstack([ F_feet[2], 0, F_feet[3] ])

    jacp_l = np.zeros((3, m.nv)) # translational Jacobian
    mujoco.mj_jacGeom(m, d, jacp_l, None, get_geom_id(m, "l_foot"))

    jacp_r = np.zeros((3, m.nv)) # translational Jacobian
    mujoco.mj_jacGeom(m, d, jacp_r, None, get_geom_id(m, "r_foot"))
    
    
    tau_l_full = jacp_l.T @ F_foot_l
    tau_r_full = jacp_r.T @ F_foot_r
    
    tau_l = tau_l_full[-4:]
    tau_r = tau_r_full[-4:]
    
    tau = (tau_l + tau_r).flatten()
    
    return tau


def get_q(d):
    return np.concatenate([
                    d.qpos,  
                    d.qvel
                    ])


def q2a():
    
    # Initial conditions: 
    # x  = 0
    # q1 = -pi/3
    # q2 =  pi/2
    # q3 = -pi/6
    # q4 =  pi/2
    
    # Set y such that "feet are slightly above the ground"
    # Note that the legs of the biped are bent with the above IC..
    # So the biped's COM z-position can safely be initialized as less than the height from its COM to its feet
    # y = 0.52
    
    m, d = load_model("hw4/assets/biped.xml")
    reset(m, d, "init")
    viewer = mujoco.viewer.launch_passive(m, d)
    camera_presets = {
                   "lookat": [0.0, 0.0, 0.55], 
                   "distance": 2, 
                   "azimuth": 90, 
                   "elevation": -10
                }  
    set_cam(viewer, track=False, presets=camera_presets, show_world_csys=False, show_body_csys=False)

    tmax = 2
    dt = m.opt.timestep
    ts = round(tmax/dt)
    data = np.zeros((ts, 7))
    time = np.arange(0, ts*dt, dt)

    for t in range(ts):

        q = get_q(d)

        u_qp = qp(q, m, d)
        d.ctrl = u_qp

        xz = np.r_[q[0], q[2]]
        theta =  R.from_quat(q[3:7], scalar_first=True).as_euler('zyx')[1:2] # rad
        data[t] = np.concatenate([xz, theta, u_qp], axis=0)
        
        mujoco.mj_step(m, d)
        viewer.sync()
        
    viewer.close()
    plot(time, data)



if __name__ == "__main__":
    q2a()