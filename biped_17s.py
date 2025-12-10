import mujoco
import numpy as np
# np.set_printoptions(precision=3)
np.set_printoptions(
    precision=3,
    suppress=True,
    formatter={
        'float_kind': lambda x: "0" if abs(x) < 1e-12 else f"{x:.3f}"
    }
)
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
solvers.options['show_progress'] = False
from utils.utils import *

class Biped():
    def __init__(self, xml, ctrl):
        
        self.m, self.d = load_model(xml)
        self.dt = self.m.opt.timestep
        
        # self.M = get_body_mass(m, "torso") + get_body_mass(m, "l_thigh") + get_body_mass(m, "l_calf") + get_body_mass(m, "r_thigh") + get_body_mass(m, "r_calf") 
        self.M = get_M(self.m)
        self.Izz = get_body_inertia(self.m, "torso")[1]
        self.g = np.abs(get_gravity(self.m))
        
        self.height = 0.69 # m
        
        self.thigh_qpos_min = -120.0*(np.pi/180) # rad
        self.thigh_qpos_max = 30.0*(np.pi/180) # rad
        self.thigh_qvel_limit = 30.0 # rad/s
        self.thigh_tau_limit = 30.0 # Nm
        
        self.calf_qpos_min = 0.0 # rad
        self.calf_qpos_max = 160.0*(np.pi/180) # rad
        self.calf_qvel_limit = 15.0 # rad/s
        self.calf_tau_limit= 60.0 # Nm
        
        self.l_foot_id = get_geom_id(self.m, "l_foot")
        self.r_foot_id = get_geom_id(self.m, "r_foot")
        self.floor_id = get_geom_id(self.m, "floor")
        
        self.foot_contact_list = [0, 0]
        self.feet_contact = False
        
        self.controllers = {
            "qp_stand": self.qp_stand,
            "pd_swing": self.pd_swing,
            "pd_step": self.pd_step,
            "qp_pd_walk": self.qp_pd_walk,
            "mpc": self.mpc,
        }
        self.ctrl = self.controllers[ctrl]
        self.u = np.array([0, 0, 0, 0])
        

        self.thigh_qpos_indices = np.array([7, 9])
        self.calf_qpos_indices = np.array([8, 10])
        
        self.thigh_qvel_indices = np.array([6, 8])
        self.calf_qvel_indices = np.array([7, 9])
        
        self.thigh_ctrl_indices = np.array([0, 2])
        self.calf_ctrl_indices = np.array([1, 3])
        

        self.l_leg_qpos_indices = np.array([7, 8])
        self.r_leg_qpos_indices = np.array([9, 10])
        self.leg_qpos_indices = np.concatenate([self.l_leg_qpos_indices, 
                                                self.r_leg_qpos_indices])
        
        self.l_leg_qvel_indices = np.array([6, 7])
        self.r_leg_qvel_indices = np.array([8, 9])
        self.leg_qvel_indices = np.concatenate([self.l_leg_qvel_indices, 
                                                self.r_leg_qvel_indices])
        
        self.l_leg_ctrl_indices = np.array([0, 1])
        self.r_leg_ctrl_indices = np.array([2, 3])
        

        # 0 = no contact, 1 = contact
        # self.stand_block = int(0.15 / self.dt)
        # self.swing_block = int(0.35 / self.dt)
        # self.gait_cycle = np.hstack([
        #         np.vstack([ np.ones((self.stand_block, 1)), np.ones((self.swing_block,  1)), np.ones((self.stand_block, 1)), np.zeros((self.swing_block, 1)) ]), # left
        #         np.vstack([ np.ones((self.stand_block, 1)), np.zeros((self.swing_block, 1)), np.ones((self.stand_block, 1)), np.ones((self.swing_block,  1)) ])  # right
        #     ])
        # self.gait_cycle_len = self.gait_cycle.shape[0]
        
        self.gait_state = 2
        self.last_gait_state = 1
        self.swing_progress = 0
        self.gait_cycle_length = int(20 / self.dt)
        
        self.l_swing_p0 = None  # left foot pose at swing start
        self.r_swing_p0 = None  # right foot pose at swing start


        self.stride_length = 0.02      # step length
        self.stride_height = 0.02      # swing height
        
        self.Fy_min = 0
        self.Fy_max = 250
        self.mu = 0.7
        
        self.first_contact = False
        
        reset(self.m, self.d, "init")
        self.step(self.u)
        
        
        
        


    def get_foot_contact(self) -> bool:
        
        self.foot_contact_list = [0, 0]
        
        for k in range(self.d.ncon):
            c = self.d.contact[k]
            g1 = c.geom1
            g2 = c.geom2
            
            for foot in (self.l_foot_id, self.r_foot_id):
                if (foot in (g1, g2)) and (self.floor_id in (g1, g2)):
                    if foot == self.l_foot_id:
                        self.foot_contact_list[0] = foot
                    elif foot == self.r_foot_id:
                        self.foot_contact_list[1] = foot
                        
        self.feet_contact = all(self.foot_contact_list)
                        

    
    def check_limits(self):
        if any(self.d.qpos[self.thigh_qpos_indices] < self.thigh_qpos_min):
            raise ValueError(f"Thigh qpos under minimum limit: {self.d.qpos[self.thigh_qpos_indices]}")

        if any(self.d.qpos[self.thigh_qpos_indices] > self.thigh_qpos_max):
            raise ValueError(f"Thigh qpos over maximum limit: {self.d.qpos[self.thigh_qpos_indices]}")

        if any(self.d.qpos[self.calf_qpos_indices] < self.calf_qpos_min):
            raise ValueError(f"Calf qpos under minimum limit: {self.d.qpos[self.calf_qpos_indices]}")

        if any(self.d.qpos[self.calf_qpos_indices] > self.calf_qpos_max):
            raise ValueError(f"Calf qpos over maximum limit: {self.d.qpos[self.calf_qpos_indices]}")

        if any(np.abs(self.d.qvel[self.thigh_qvel_indices]) > self.thigh_qvel_limit):
            raise ValueError(f"Thigh qvel exceeds limit: {self.d.qvel[self.thigh_qvel_indices]}")

        if any(np.abs(self.d.qvel[self.calf_qvel_indices]) > self.calf_qvel_limit):
            raise ValueError(f"Calf qvel exceeds limit: {self.d.qvel[self.calf_qvel_indices]}")

        if any(np.abs(self.d.ctrl[self.thigh_ctrl_indices]) > self.thigh_tau_limit):
            raise ValueError(f"Thigh ctrl exceeds limit: {self.d.ctrl[self.thigh_ctrl_indices]}")

        if any(np.abs(self.d.ctrl[self.calf_ctrl_indices]) > self.calf_tau_limit):
            raise ValueError(f"Calf ctrl exceeds limit: {self.d.ctrl[self.calf_ctrl_indices]}")



            

    def set_tau_limits(self, u):
        
        u[[0, 2]] = np.clip(u[[0, 2]], -self.thigh_tau_limit, self.thigh_tau_limit)
        u[[1, 3]] = np.clip(u[[1, 3]], -self.calf_tau_limit, self.calf_tau_limit)
        
        return u



    def step(self, u):
        self.get_foot_contact()
        
        if not self.first_contact and self.feet_contact:
            self.first_contact = True
        
        self.u = u #self.set_tau_limits(u)
        
        self.d.ctrl = self.u
        # self.check_limits()




                        
    
    def qp_stand(self, xf):
        
        if not self.feet_contact: return np.zeros(self.m.nu)
        
        alpha = 0.001
        Q = np.array([
            [1,    0,  0.1],
            [0,    1,    0],
            [0.1,  0,    5],
        ])
        
        xc = self.d.qpos[0]
        yc = self.d.qpos[2]
        theta_c_quat = self.d.qpos[3:7]
        theta_c = R.from_quat(theta_c_quat, scalar_first=True).as_euler('zyx')[1]
        
        xc_dot = self.d.qvel[0]
        yc_dot = self.d.qvel[2]
        theta_c_dot = self.d.qvel[4]
        
        xc_des, yc_des, theta_c_des, xc_dot_des, yc_dot_des, theta_c_dot_des = xf
               
        Kp_x = 5
        Kd_x = 10
        Kp_y = 5
        Kd_y = 10
        Kp_theta = 0.5 
        Kd_theta = 0.5 

        x_ddot_des = Kp_x * (xc_des - xc) + Kd_x * (xc_dot_des - xc_dot)
        y_ddot_des = Kp_y * (yc_des - yc) + Kd_y * (yc_dot_des - yc_dot)
        theta_ddot_des = Kp_theta * (theta_c_des - theta_c) + Kd_theta * (theta_c_dot_des - theta_c_dot)
        
        PF1, PF2 = get_feet_xpos(self.m, self.d)[:, :]
        PF1x, PF1y = PF1[[0, 2]]
        PF2x, PF2y = PF2[[0, 2]]
        
        A = np.array([
            [1,       0,       1,       0      ],
            [0,       1,       0,       1      ],
            [yc-PF1y, PF1x-xc, yc-PF2y, PF2x-xc]
        ])
        
        b = np.array([
            [self.M * x_ddot_des],
            [self.M * (y_ddot_des + self.g)],
            [self.Izz * theta_ddot_des],
        ])
        
        n = A.shape[1]
        
        P = 2 * (A.T @ Q @ A + alpha * np.eye(n))
        q = -2 * A.T @ Q @ b        
        G = np.array([
            [0,         1,   0,        0],
            [0,         0,   0,        1],
            [0,        -1,   0,        0],
            [0,         0,   0,       -1],
            [1,  -self.mu,   0,        0],
            [0,         0,   1, -self.mu],
            [-1, -self.mu,   0,        0],
            [0,         0,  -1, -self.mu]
        ])
        h = np.array([
            [self.Fy_max],
            [self.Fy_max],
            [-self.Fy_min],
            [-self.Fy_min],
            [0],
            [0],
            [0],
            [0]
        ])
        
        P_cvx = matrix(P.astype(float))
        q_cvx = matrix(q.astype(float))
        G_cvx = matrix(G.astype(float))
        h_cvx = matrix(h.astype(float))
        
        sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
        F_GRF_star = np.array([sol['x']])[0]
        
        F_feet = -F_GRF_star
        
        F_foot_l = np.vstack([ F_feet[0], 0, F_feet[1] ])
        F_foot_r = np.vstack([ F_feet[2], 0, F_feet[3] ])

        jacp_l = np.zeros((3, self.m.nv))
        mujoco.mj_jacGeom(self.m, self.d, jacp_l, None, self.l_foot_id)

        jacp_r = np.zeros((3, self.m.nv))
        mujoco.mj_jacGeom(self.m, self.d, jacp_r, None, self.r_foot_id)
        
        tau_l_full = jacp_l.T @ F_foot_l
        tau_r_full = jacp_r.T @ F_foot_r
        
        tau_l = tau_l_full[-4:]
        tau_r = tau_r_full[-4:]
        
        tau = (tau_l + tau_r).flatten()
        
        return tau
  

    def qp_step(self, xf, stance):
        """gpt"""
        # if not self.feet_contact: return np.zeros(self.m.nu)

        alpha = 0.0001
        Q = np.array([
            [10,    0,  0.1],
            [0,    1,    0],
            [0.1,  0,    10],
        ])

        # current centroidal state
        xc = self.d.qpos[0]
        yc = self.d.qpos[2]
        theta_c_quat = self.d.qpos[3:7]
        theta_c = R.from_quat(theta_c_quat, scalar_first=True).as_euler('zyx')[1]

        xc_dot = self.d.qvel[0]
        yc_dot = self.d.qvel[2]
        theta_c_dot = self.d.qvel[4]

        # desired centroidal state (passed in)
        xc_des, yc_des, theta_c_des, xc_dot_des, yc_dot_des, theta_c_dot_des = xf

        # moderate gains, not insane stiff
        Kp_x = 5.0
        Kd_x = 3.0
        Kp_y = 5.0
        Kd_y = 3.0
        Kp_theta = 0.5
        Kd_theta = 0.5

        x_ddot_des = Kp_x * (xc_des - xc) + Kd_x * (xc_dot_des - xc_dot)
        y_ddot_des = Kp_y * (yc_des - yc) + Kd_y * (yc_dot_des - yc_dot)
        theta_ddot_des = Kp_theta * (theta_c_des - theta_c) + Kd_theta * (theta_c_dot_des - theta_c_dot)

        # foot positions
        PF1, PF2 = get_feet_xpos(self.m, self.d)[:, :]
        PF1x, PF1y = PF1[[0, 2]]
        PF2x, PF2y = PF2[[0, 2]]

        # centroidal dynamics matrix
        A = np.array([
            [1,       0,       1,       0      ],
            [0,       1,       0,       1      ],
            [yc-PF1y, PF1x-xc, yc-PF2y, PF2x-xc]
        ])

        b = np.array([
            [self.M * x_ddot_des],
            [self.M * (y_ddot_des + self.g)],
            [self.Izz * theta_ddot_des],
        ])

        n = A.shape[1]

        P = 2 * (A.T @ Q @ A + alpha * np.eye(n))
        q = -2 * (A.T @ Q @ b)

        # normal force bounds per foot
        Fy_max_L = self.Fy_max
        Fy_max_R = self.Fy_max
        Fy_min_L = self.Fy_min
        Fy_min_R = self.Fy_min

        # encode single support by killing the swing foot inside the QP
        if stance == "left":
            # right foot must have zero normal force
            Fy_max_R = 0.0
            Fy_min_R = 0.0
        elif stance == "right":
            # left foot must have zero normal force
            Fy_max_L = 0.0
            Fy_min_L = 0.0
        # stance == "both" keeps both feet active

        # inequalities G F <= h
        # 1: Fy_L <= Fy_max_L
        # 2: Fy_R <= Fy_max_R
        # 3: -Fy_L <= -Fy_min_L  -> Fy_L >= Fy_min_L
        # 4: -Fy_R <= -Fy_min_R  -> Fy_R >= Fy_min_R
        # 5,6,7,8: friction cones |Fx| <= mu * Fy for each foot
        G = np.array([
            [0,         1,   0,        0],
            [0,         0,   0,        1],
            [0,        -1,   0,        0],
            [0,         0,   0,       -1],
            [1,  -self.mu,   0,        0],
            [0,         0,   1, -self.mu],
            [-1, -self.mu,   0,        0],
            [0,         0,  -1, -self.mu]
        ])

        h = np.array([
            [Fy_max_L],
            [Fy_max_R],
            [-Fy_min_L],
            [-Fy_min_R],
            [0],
            [0],
            [0],
            [0]
        ])

        P_cvx = matrix(P.astype(float))
        q_cvx = matrix(q.astype(float))
        G_cvx = matrix(G.astype(float))
        h_cvx = matrix(h.astype(float))

        sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
        F_GRF_star = np.array([sol['x']])[0]

        # sign convention, match your original code
        F_feet = -F_GRF_star

        # map planar forces back to 3D for each foot
        F_foot_l = np.vstack([F_feet[0], 0, F_feet[1]])
        F_foot_r = np.vstack([F_feet[2], 0, F_feet[3]])

        # foot Jacobians
        jacp_l = np.zeros((3, self.m.nv))
        mujoco.mj_jacGeom(self.m, self.d, jacp_l, None, self.l_foot_id)

        jacp_r = np.zeros((3, self.m.nv))
        mujoco.mj_jacGeom(self.m, self.d, jacp_r, None, self.r_foot_id)

        tau_l_full = jacp_l.T @ F_foot_l
        tau_r_full = jacp_r.T @ F_foot_r

        # last 4 entries are the actuated joints on each leg
        tau_l = tau_l_full[-4:]
        tau_r = tau_r_full[-4:]

        tau = (tau_l + tau_r).flatten()

        return tau

    def pd_swing(self):
        
        amp_x = 0.05
        amp_z = 0.05
        freq = 0.5
        w = 2 * np.pi * freq 
        
        x_offset = 0.05
        z_offset = 0.05
        phase_shfit = np.pi
        
        kp = np.diag([5.0, 5.0])
        kd = np.diag([1.0, 1.0])
        
        damping = 0.01
        regularization = damping**2 * np.eye(3)
        
        xpos_feet = get_feet_xpos(self.m, self.d)
        t = self.d.time
        
        x_des_l = amp_x * np.sin(w * t) + x_offset
        y_des_l = xpos_feet[0][1]
        z_des_l = amp_z * np.sin(w * t) + z_offset
        xpos_des_l = np.array([x_des_l, y_des_l, z_des_l])
        xpos_err_l = xpos_des_l - xpos_feet[0]

        x_des_r = amp_x * np.sin(w * t - phase_shfit) + x_offset
        y_des_r = xpos_feet[1][1]
        z_des_r = amp_z * np.sin(w * t - phase_shfit) + z_offset
        xpos_des_r = np.array([x_des_r, y_des_r, z_des_r])
        xpos_err_r = xpos_des_r - xpos_feet[1]

        jacp_l_full = np.zeros((3, self.m.nv))
        mujoco.mj_jacGeom(self.m, self.d, jacp_l_full, None, self.l_foot_id)
        jacp_l = jacp_l_full[:, self.l_leg_qvel_indices]

        jacp_r_full = np.zeros((3, self.m.nv))
        mujoco.mj_jacGeom(self.m, self.d, jacp_r_full, None, self.r_foot_id)
        jacp_r = jacp_r_full[:, self.r_leg_qvel_indices]
                
        # regularized pseudoinverse
        jacp_lT = jacp_l @ jacp_l.T
        jacp_l_pinv = jacp_l.T @ np.linalg.inv(jacp_lT + regularization)
        
        jacp_rT = jacp_r @ jacp_r.T
        jacp_r_pinv = jacp_r.T @ np.linalg.inv(jacp_rT + regularization)
        
        qpos_err_l = jacp_l_pinv @ xpos_err_l
        qpos_err_r = jacp_r_pinv @ xpos_err_r
        
        qvel_err_l = -self.d.qvel[self.l_leg_qvel_indices]
        qvel_err_r = -self.d.qvel[self.r_leg_qvel_indices]
        
        u_l = kp @ qpos_err_l + kd @ qvel_err_l
        u_r = kp @ qpos_err_r + kd @ qvel_err_r
            
        return np.concatenate([u_l, u_r])
    
    def pd_step(self, leg, phi):
        """
        Swing-leg operational-space PD using Jacobian transpose.

        Parameters
        ----------
        leg : str
            "left" or "right".
        phi : float
            Phase in [0, 1] within the current swing cycle.
        """

        
        xpos_feet = get_feet_xpos(self.m, self.d)
        theta_c = R.from_quat(self.d.qpos[3:7], scalar_first=True).as_euler('zyx')[1]
        
        opt_sep = 0.3
        curr_sep = xpos_feet[1][0] - xpos_feet[0][0]

        # Cartesian stiffness and damping (N/m and N·s/m)
        # gain_left = 13 * (1-theta_c) * (curr_sep-opt_sep)
        # gain_right = 14 * (1+theta_c) * (1-(curr_sep-opt_sep))
        gain_left = 1 * (1+(curr_sep-opt_sep))
        gain_right = 10000 * (1-(curr_sep-opt_sep))
        
        kp_val_left = 5.0
        kd_val_left = 2.0
        Kp_left = np.diag([kp_val_left, kp_val_left, kp_val_left])
        Kd_left = np.diag([kd_val_left, kd_val_left, kd_val_left])
        
        kp_val_right = 10.0
        kd_val_right = 2.0
        Kp_right = np.diag([kp_val_right, kp_val_right, kp_val_right])
        Kd_right = np.diag([kd_val_right, kd_val_right, kd_val_right])

        # Desired swing-foot trajectory in world coordinates
        if leg == "left":
            p0 = self.l_swing_p0 if self.l_swing_p0 is not None else xpos_feet[0]
            p1 = p0 + gain_left*np.array([self.stride_length, 0.0, 0.0])

            x_des = 0.8 * ((1.0 - phi) * p0[0] + phi * p1[0])
            y_des = p0[1]
            z_des = p0[2] + self.stride_height * 1 * np.sin(np.pi * phi) #+ 2.5
            xpos_des = np.array([x_des, y_des, z_des])
            xpos_curr = xpos_feet[0]

            jacp_full = np.zeros((3, self.m.nv))
            mujoco.mj_jacGeom(self.m, self.d, jacp_full, None, self.l_foot_id)
            jac = jacp_full[:, self.l_leg_qvel_indices]
            qvel = self.d.qvel[self.l_leg_qvel_indices]

        elif leg == "right":
            p0 = self.r_swing_p0 if self.r_swing_p0 is not None else xpos_feet[1]
            p1 = p0 + gain_right*np.array([self.stride_length, 0.0, 0.0])

            x_des = ((1.0 - phi) * p0[0] + phi * p1[0])
            y_des = p0[1]
            z_des = p0[2] + self.stride_height * 10 * np.sin(np.pi * phi) + 5
            xpos_des = np.array([x_des, y_des, z_des])
            xpos_curr = xpos_feet[1]

            jacp_full = np.zeros((3, self.m.nv))
            mujoco.mj_jacGeom(self.m, self.d, jacp_full, None, self.r_foot_id)
            jac = jacp_full[:, self.r_leg_qvel_indices]
            qvel = self.d.qvel[self.r_leg_qvel_indices]

        # if leg == "right":
            # print(f"gait_state: {self.gait_state}, curr_sep: {curr_sep :.3f}, theta_c: {theta_c :.3f}, phi: {phi :.3f}, foot: {leg}, x_curr: {xpos_curr}, x_des: {xpos_des}, p0: {p0}, p1: {p1}")
        # print(f"gl: {gain_left :.3f}, gr: {gain_right :.3f}, sep: {curr_sep :.3f}")
    
        # Cartesian spring-damper: F = Kp * x_err - Kd * v_curr
        x_err = xpos_des - xpos_curr
        v_curr = jac @ qvel
        
        Kp = Kp_left if leg == "left" else Kp_right
        Kd = Kd_left if leg == "left" else Kd_right
        
        F_cart = Kp @ x_err - Kd @ v_curr

        # Map Cartesian force to joint torques: tau = J^T * F
        tau_leg = jac.T @ F_cart

        # Construct full actuator vector [l_thigh, l_calf, r_thigh, r_calf]
        if leg == "left":
            u_l = tau_leg
            u_r = np.zeros_like(tau_leg)
        else:
            u_l = np.zeros_like(tau_leg)
            u_r = tau_leg

        return np.concatenate([u_l, u_r])
    


    def qp_pd_walk(self):
        """
        QP + PD walking controller with torso-angle safety.

        gait_state:
        0: right swing, left stance
        1: left swing, right stance
        2: double-support safety / recovery
        """
        
        if not self.first_contact: return np.zeros(self.m.nu)
        
        # 1. Torso pitch from base quaternion
        theta_c_quat = self.d.qpos[3:7]
        theta_c = R.from_quat(theta_c_quat, scalar_first=True).as_euler('zyx')[1]
        theta_c_vel = self.d.qvel[4]

        
        xcm = get_x_com(self.m, self.d)[0]
        print(f"gait: {self.gait_state}, last_gait: {self.last_gait_state}, theta_c: {theta_c :.3f}, q4: {self.d.qpos[-1:][0]:.3f}, x_cm: {xcm :.3f}, t: {self.d.time :.3f}")
        if xcm >= 1:
            print(f"finished, {self.d.time}")
            exit()
        
        planted = self.d.qpos[-1] >= 0.2 #0.4  # 0.468

        # 2. Safety hysteresis thresholds [rad]
        theta_enter = 0.1   # enter safety if |theta| > 0.20
        theta_exit  = 0.05   # leave safety only if |theta| < 0.10
        # theta_enter = 0.1   # enter safety if |theta| > 0.20
        # theta_exit  = 0.05   # leave safety only if |theta| < 0.10
        theta_vel_exit  = 0.05   # leave safety only if |theta| < 0.10

        # 3. Update safety mode
        if self.gait_state == 2:
            # Already in safety, leave only if back in tighter band
            # if np.abs(theta_c) < theta_exit and theta_c_vel < theta_vel_exit:
            #     self.gait_state = 0
            #     self.swing_progress = 0

            # if np.abs(theta_c) < theta_exit and planted: #and theta_c_vel < theta_vel_exit:
            if np.abs(theta_c) < theta_exit and ( (planted and self.last_gait_state == 1) or self.last_gait_state == 0 ):
                if self.last_gait_state == 1:
                    self.gait_state = 0
                elif self.last_gait_state == 0:
                    self.gait_state = 1
                self.swing_progress = 0
        else:
            # Not in safety, enter if exceeded large limit
            if np.abs(theta_c) > theta_enter: #and theta_c_vel > theta_vel_exit:
                if self.swing_progress < 0.5 * self.gait_cycle_length:
                    self.last_gait_state = self.gait_state
                    self.gait_state = 2
                    self.swing_progress = 0

        # 4. Safety mode: stand in double support and keep torso upright
        if self.gait_state == 2:
            xpos_com = get_x_com(self.m, self.d)
            xc_des = xpos_com[0]
            yc_des = self.height - 0.2
            theta_c_des = 0
            xcdot_des = 0.01
            ycdot_des = 0
            theta_c_dot_des = 0
            xf_stand = np.array([xc_des, yc_des, theta_c_des, xcdot_des, ycdot_des, theta_c_dot_des])
            tau = self.qp_step(xf_stand, stance="both")
            # Debug if you want:
            # print(f"SAFETY theta_c: {theta_c:.3f}, gait_state: {self.gait_state:d}")
            return tau

        # 5. Normal gait state transitions
        if self.swing_progress >= self.gait_cycle_length:
            self.swing_progress = 0
            # if self.gait_state == 0:
            #     self.gait_state = 1   # switch to left swing
            # else:
            #     self.gait_state = 0   # switch to right swing
            if self.gait_state == 0:
                self.gait_state = 1   # switch to left swing
            else:
                self.gait_state = 0   # switch to right swing

        # 6. Compute stance and swing torques
        xpos_feet = get_feet_xpos(self.m, self.d)
        
        xpos_com = get_x_com(self.m, self.d)
        xc_des = xpos_com[0] #+ self.stride_length
        yc_des = self.height - 0.5
        theta_c_des = 50
        xcdot_des = -0.5
        ycdot_des = -0.85
        theta_c_dot_des = 5000
        xf_step = np.array([xc_des, yc_des, theta_c_des, xcdot_des, ycdot_des, theta_c_dot_des])

        if self.gait_state == 0:
            # Right swing, left stance
            if self.swing_progress == 0:
                # Store right swing start pose
                self.r_swing_p0 = xpos_feet[1].copy()

            phase = self.swing_progress / self.gait_cycle_length
            self.swing_progress += 1

            # Stance leg via QP
            tau = self.qp_step(xf_step, stance="left")
            # Swing leg via PD foot tracking
            tau_swing = self.pd_step("right", phase)
            tau[self.r_leg_ctrl_indices] = tau_swing[self.r_leg_ctrl_indices]

        elif self.gait_state == 1:
            # Left swing, right stance
            if self.swing_progress == 0:
                # Store left swing start pose
                self.l_swing_p0 = xpos_feet[0].copy()

            phase = self.swing_progress / self.gait_cycle_length
            self.swing_progress += 1

            # Stance leg via QP
            tau = self.qp_step(xf_step, stance="right")
            # Swing leg via PD foot tracking
            tau_swing = self.pd_step("left", phase)
            tau[self.l_leg_ctrl_indices] = tau_swing[self.l_leg_ctrl_indices]

        return tau
    

  
    # def qp_step(self, xf, stance):
                
    #     alpha = 0.001
    #     Q = np.array([
    #         [1,    0,  0.1],
    #         [0,    1,    0],
    #         [0.1,  0,    5],
    #     ])
        
    #     xc = self.d.qpos[0]
    #     yc = self.d.qpos[2]
    #     theta_c_quat = self.d.qpos[3:7]
    #     theta_c = R.from_quat(theta_c_quat, scalar_first=True).as_euler('zyx')[1]
        
    #     xc_dot = self.d.qvel[0]
    #     yc_dot = self.d.qvel[2]
    #     theta_c_dot = self.d.qvel[4]
        
    #     xc_des, yc_des, theta_c_des, xc_dot_des, yc_dot_des, theta_c_dot_des = xf
               
    #     Kp_x = 5
    #     Kd_x = 10
    #     Kp_y = 5
    #     Kd_y = 10
    #     Kp_theta = 0.5 
    #     Kd_theta = 0.5 

    #     x_ddot_des = Kp_x * (xc_des - xc) + Kd_x * (xc_dot_des - xc_dot)
    #     y_ddot_des = Kp_y * (yc_des - yc) + Kd_y * (yc_dot_des - yc_dot)
    #     theta_ddot_des = Kp_theta * (theta_c_des - theta_c) + Kd_theta * (theta_c_dot_des - theta_c_dot)
        
    #     PF1, PF2 = get_feet_xpos(self.m, self.d)[:, :]
    #     PF1x, PF1y = PF1[[0, 2]]
    #     PF2x, PF2y = PF2[[0, 2]]
        
    #     A = np.array([
    #         [1,       0,       1,       0      ],
    #         [0,       1,       0,       1      ],
    #         [yc-PF1y, PF1x-xc, yc-PF2y, PF2x-xc]
    #     ])
        
    #     b = np.array([
    #         [self.M * x_ddot_des],
    #         [self.M * (y_ddot_des + self.g)],
    #         [self.Izz * theta_ddot_des],
    #     ])
        
    #     n = A.shape[1]
        
    #     P = 2 * (A.T @ Q @ A + alpha * np.eye(n))
    #     q = -2 * A.T @ Q @ b        
    #     G = np.array([
    #         [0,         1,   0,        0],
    #         [0,         0,   0,        1],
    #         [0,        -1,   0,        0],
    #         [0,         0,   0,       -1],
    #         [1,  -self.mu,   0,        0],
    #         [0,         0,   1, -self.mu],
    #         [-1, -self.mu,   0,        0],
    #         [0,         0,  -1, -self.mu]
    #     ])
    #     h = np.array([
    #         [self.Fy_max],
    #         [self.Fy_max],
    #         [-self.Fy_min],
    #         [-self.Fy_min],
    #         [0],
    #         [0],
    #         [0],
    #         [0]
    #     ])
        
    #     P_cvx = matrix(P.astype(float))
    #     q_cvx = matrix(q.astype(float))
    #     G_cvx = matrix(G.astype(float))
    #     h_cvx = matrix(h.astype(float))
        
    #     sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    #     F_GRF_star = np.array([sol['x']])[0]
        
    #     F_feet = -F_GRF_star
        
    #     if stance == "left":
    #         F_feet[2] = 0.0
    #         F_feet[3] = 0.0
    #     elif stance == "right":
    #         F_feet[0] = 0.0
    #         F_feet[1] = 0.0
        
    #     F_foot_l = np.vstack([ F_feet[0], 0, F_feet[1] ])
    #     F_foot_r = np.vstack([ F_feet[2], 0, F_feet[3] ])

    #     jacp_l = np.zeros((3, self.m.nv))
    #     mujoco.mj_jacGeom(self.m, self.d, jacp_l, None, self.l_foot_id)

    #     jacp_r = np.zeros((3, self.m.nv))
    #     mujoco.mj_jacGeom(self.m, self.d, jacp_r, None, self.r_foot_id)
        
    #     tau_l_full = jacp_l.T @ F_foot_l
    #     tau_r_full = jacp_r.T @ F_foot_r
        
    #     tau_l = tau_l_full[-4:]
    #     tau_r = tau_r_full[-4:]
        
    #     tau = (tau_l + tau_r).flatten()
        
    #     return tau
    

    

    # def qp_step(self, xf, stance):
    #     "gemini"
    #     # UNPACK TARGETS: Use the xf passed in, do not hardcode 0!
    #     xc_des, yc_des, theta_c_des, xc_dot_des, yc_dot_des, theta_c_dot_des = xf
        
    #     # --- Standard State Feedback ---
    #     alpha = 0.001
    #     Q = np.array([
    #         [1,    0,  0.1],
    #         [0,    1,    0],
    #         [0.1,  0,    5],
    #     ])
        
    #     xc = self.d.qpos[0]
    #     yc = self.d.qpos[2]
    #     theta_c_quat = self.d.qpos[3:7]
    #     theta_c = R.from_quat(theta_c_quat, scalar_first=True).as_euler('zyx')[1]
        
    #     xc_dot = self.d.qvel[0]
    #     yc_dot = self.d.qvel[2]
    #     theta_c_dot = self.d.qvel[4]
        
    #     Kp_x = 5
    #     Kd_x = 10
    #     Kp_y = 5
    #     Kd_y = 10
    #     Kp_theta = 500
    #     Kd_theta = 10 

    #     x_ddot_des = Kp_x * (xc_des - xc) + Kd_x * (xc_dot_des - xc_dot)
    #     y_ddot_des = Kp_y * (yc_des - yc) + Kd_y * (yc_dot_des - yc_dot)
    #     theta_ddot_des = Kp_theta * (theta_c_des - theta_c) + Kd_theta * (theta_c_dot_des - theta_c_dot)
        
    #     PF1, PF2 = get_feet_xpos(self.m, self.d)[:, :]
    #     PF1x, PF1y = PF1[[0, 2]]
    #     PF2x, PF2y = PF2[[0, 2]]
        
    #     A = np.array([
    #         [1,       0,       1,       0      ],
    #         [0,       1,       0,       1      ],
    #         [yc-PF1y, PF1x-xc, yc-PF2y, PF2x-xc]
    #     ])
        
    #     b = np.array([
    #         [self.M * x_ddot_des],
    #         [self.M * (y_ddot_des + self.g)],
    #         [self.Izz * theta_ddot_des],
    #     ])
        
    #     n = A.shape[1]
        
    #     P = 2 * (A.T @ Q @ A + alpha * np.eye(n))
    #     q = -2 * A.T @ Q @ b        
        
    #     # --- CONSTRAINT FIX: Enforce Single Support in the QP ---
    #     Fy_max_L = self.Fy_max
    #     Fy_max_R = self.Fy_max
    #     Fy_min_L = self.Fy_min
    #     Fy_min_R = self.Fy_min
        
    #     # If left stance, Right foot must have 0 force
    #     if stance == "left":
    #         Fy_max_R = 0.0
    #         Fy_min_R = 0.0
    #     # If right stance, Left foot must have 0 force
    #     elif stance == "right":
    #         Fy_max_L = 0.0
    #         Fy_min_L = 0.0

    #     G = np.array([
    #         [0,         1,   0,        0], # Fy_L <= Max
    #         [0,         0,   0,        1], # Fy_R <= Max
    #         [0,        -1,   0,        0], # -Fy_L <= -Min -> Fy_L >= Min
    #         [0,         0,   0,       -1], # -Fy_R <= -Min -> Fy_R >= Min
    #         [1,  -self.mu,   0,        0],
    #         [0,         0,   1, -self.mu],
    #         [-1, -self.mu,   0,        0],
    #         [0,         0,  -1, -self.mu]
    #     ])
    #     h = np.array([
    #         [Fy_max_L],
    #         [Fy_max_R],
    #         [-Fy_min_L],
    #         [-Fy_min_R],
    #         [0],
    #         [0],
    #         [0],
    #         [0]
    #     ])
        
    #     P_cvx = matrix(P.astype(float))
    #     q_cvx = matrix(q.astype(float))
    #     G_cvx = matrix(G.astype(float))
    #     h_cvx = matrix(h.astype(float))
        
    #     sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    #     F_GRF_star = np.array([sol['x']])[0]
        
    #     F_feet = -F_GRF_star
        
    #     # DO NOT manually zero out forces here. The QP constraints above handled it.
    #     # If you zero them here, you invalidate the QP solution.
        
    #     F_foot_l = np.vstack([ F_feet[0], 0, F_feet[1] ])
    #     F_foot_r = np.vstack([ F_feet[2], 0, F_feet[3] ])

    #     jacp_l = np.zeros((3, self.m.nv))
    #     mujoco.mj_jacGeom(self.m, self.d, jacp_l, None, self.l_foot_id)

    #     jacp_r = np.zeros((3, self.m.nv))
    #     mujoco.mj_jacGeom(self.m, self.d, jacp_r, None, self.r_foot_id)
        
    #     tau_l_full = jacp_l.T @ F_foot_l
    #     tau_r_full = jacp_r.T @ F_foot_r
        
    #     tau_l = tau_l_full[-4:]
    #     tau_r = tau_r_full[-4:]
        
    #     tau = (tau_l + tau_r).flatten()
        
    #     return tau

    

    # def pd_step(self, leg, phi):
                
    #     kp_l = np.diag([2.0, 2.0])
    #     kd_l = np.diag([1.0, 1.0])
        
    #     kp_r = np.diag([5.0, 5.0])
    #     kd_r = np.diag([1.0, 1.0])
        
    #     damping = 0.01
    #     regularization = damping**2 * np.eye(3)
        
    #     xpos_feet = get_feet_xpos(self.m, self.d)

    #     # phi in [0, 1] passed in from gait logic

    #     xpos_feet = get_feet_xpos(self.m, self.d)

    #     if leg == "left":
    #         p0 = self.l_swing_p0 if self.l_swing_p0 is not None else xpos_feet[0]
    #         p1 = p0 + np.array([self.stride_length, 0.0, 0.0])     # forward step target

    #         x_des = (1.0 - phi) * p0[0] + phi * p1[0]
    #         y_des = p0[1]
    #         z_des = p0[2] + self.stride_height * np.sin(np.pi * phi)

    #         xpos_des_l = np.array([x_des, y_des, z_des])
    #         xpos_des_r = xpos_feet[1]

    #     elif leg == "right":
    #         p0 = self.r_swing_p0 if self.r_swing_p0 is not None else xpos_feet[1]
    #         p1 = p0 + np.array([self.stride_length, 0.0, 0.0])     # forward step target

    #         x_des = (1.0 - phi) * p0[0] + phi * p1[0]
    #         y_des = p0[1]
    #         z_des = p0[2] + self.stride_height * np.sin(np.pi * phi)

    #         xpos_des_r = np.array([x_des, y_des, z_des])
    #         xpos_des_l = xpos_feet[0]

    #     # print(self.r_swing_p0, p0, p1, x_step, x_des, phi)
    #     print(f"t: {self.d.time :.3f}, leg: {leg}, xl_curr: {xpos_feet[0]}, xl_des: {xpos_des_l}, xr_curr: {xpos_feet[1]}, xr_des: {xpos_des_r}")
    #     # exit()
        
    #     xpos_err_l = xpos_des_l - xpos_feet[0]
    #     xpos_err_r = xpos_des_r - xpos_feet[1]

    #     jacp_l_full = np.zeros((3, self.m.nv))
    #     mujoco.mj_jacGeom(self.m, self.d, jacp_l_full, None, self.l_foot_id)
    #     jacp_l = jacp_l_full[:, self.l_leg_qvel_indices]

    #     jacp_r_full = np.zeros((3, self.m.nv))
    #     mujoco.mj_jacGeom(self.m, self.d, jacp_r_full, None, self.r_foot_id)
    #     jacp_r = jacp_r_full[:, self.r_leg_qvel_indices]
                
    #     # regularized pseudoinverse
    #     jacp_lT = jacp_l @ jacp_l.T
    #     jacp_l_pinv = jacp_l.T @ np.linalg.inv(jacp_lT + regularization)
        
    #     jacp_rT = jacp_r @ jacp_r.T
    #     jacp_r_pinv = jacp_r.T @ np.linalg.inv(jacp_rT + regularization)
        
    #     qpos_err_l = jacp_l_pinv @ xpos_err_l
    #     qpos_err_r = jacp_r_pinv @ xpos_err_r
        
    #     qvel_err_l = -self.d.qvel[self.l_leg_qvel_indices]
    #     qvel_err_r = -self.d.qvel[self.r_leg_qvel_indices]
        
    #     u_l = kp_l @ qpos_err_l + kd_l @ qvel_err_l
    #     u_r = kp_r @ qpos_err_r + kd_r @ qvel_err_r
            
    #     return np.concatenate([u_l, u_r])


    # def qp_pd_walk(self, t, xf):
        
    #     t_gait = t % self.gait_cycle_len
    #     gait = self.gait_cycle[t_gait]

    #     theta_c_quat = self.d.qpos[3:7]
    #     theta_c = R.from_quat(theta_c_quat, scalar_first=True).as_euler('zyx')[1]
    #     print(theta_c)
        
    #     if np.abs(theta_c) > 0.1:
    #         tau = self.qp_step(xf, stance="both") # double support

                
    #     elif gait[0] == 0 and gait[1] == 1:
    #         # print("left swing")                   
    #         l_swing_start = 2*self.stand_block + self.swing_block
    #         phase = (t_gait - l_swing_start) / self.swing_block

    #         if t_gait == l_swing_start:
    #             xpos_feet = get_feet_xpos(self.m, self.d)
    #             self.l_swing_p0 = xpos_feet[0].copy()
                
    #         tau = self.qp_step(xf, stance="right") # left swing => right stance
    #         tau_swing = self.pd_step("left", phase)
    #         tau[self.l_leg_ctrl_indices] = tau_swing[self.l_leg_ctrl_indices]
            
    #     elif gait[0] == 1 and gait[1] == 0:
    #         # print("right swing")                   
    #         r_swing_start = self.stand_block
    #         phase = (t_gait - r_swing_start) / self.swing_block

    #         if t_gait == r_swing_start:
    #             xpos_feet = get_feet_xpos(self.m, self.d)
    #             self.r_swing_p0 = xpos_feet[1].copy()
                
    #         tau = self.qp_step(xf, stance="left") # right swing => left stance
    #         tau_swing = self.pd_step("right", phase)
    #         tau[self.r_leg_ctrl_indices] = tau_swing[self.r_leg_ctrl_indices]
            
    #     else:               
    #         print("both")                   
    #         tau = self.qp_step(xf, stance="both") # double support
            
    #     return tau
    
    # def qp_pd_walk(self, xf):
        
    #     theta_c = R.from_quat(self.d.qpos[3:7], scalar_first=True).as_euler('zyx')[1]
    #     if np.abs(theta_c) > 0.1246:
    #         # print("both")
    #         self.gait_state = 2
    #         tau = self.qp_step(xf, stance="both") # double support
    #         print(f"theta_c: {theta_c :.3f}, gait_state: {self.gait_state :.0f}, swing_prog: {self.swing_progress :.0f}, phase: {0.000 :.3f}, cycle_len: {self.gait_cycle_length :.0f}")
    #         return tau
        
    #     # theta_c_vel = self.d.qvel[4]
    #     # if np.abs(theta_c_vel) > 0.3:
    #     #     # print("both")
    #     #     self.gait_state = 2
    #     #     tau = self.qp_step(xf, stance="both") # double support
    #     #     print(f"theta_c_vel: {theta_c_vel :.3f}, gait_state: {self.gait_state :.0f}, swing_prog: {self.swing_progress :.0f}, phase: {0.000 :.3f}, cycle_len: {self.gait_cycle_length :.0f}")
    #     #     return tau
            
    #     if self.gait_state == 2:
    #         self.gait_state = 0
    #         self.swing_progress = 0
    #     elif self.swing_progress >= self.gait_cycle_length:
    #         self.swing_progress = 0
    #         if self.gait_state == 0: self.gait_state = 1
    #         else: self.gait_state = 0
            
     
    #     xpos_feet = get_feet_xpos(self.m, self.d)

    #     if self.gait_state == 0:
    #         # print("right swing")

    #         if self.swing_progress == 0:
    #             self.r_swing_p0 = xpos_feet[1].copy()
                
    #         phase = self.swing_progress / self.gait_cycle_length
    #         self.swing_progress += 1                   
                
    #         tau = self.qp_step(xf, stance="left") # right swing => left stance
    #         tau_swing = self.pd_step("right", phase)
    #         tau[self.r_leg_ctrl_indices] = tau_swing[self.r_leg_ctrl_indices]
            
    #     elif self.gait_state == 1:
    #         # print("left swing")                   

    #         if self.swing_progress == 0:
    #             self.l_swing_p0 = xpos_feet[0].copy()
                
    #         phase = self.swing_progress / self.gait_cycle_length
    #         self.swing_progress += 1                   
            
    #         tau = self.qp_step(xf, stance="right") # left swing => right stance
    #         tau_swing = self.pd_step("left", phase)
    #         tau[self.l_leg_ctrl_indices] = tau_swing[self.l_leg_ctrl_indices]
            

    #     print(f"theta_c: {theta_c :.3f}, gait_state: {self.gait_state :.0f}, swing_prog: {self.swing_progress :.0f}, phase: {phase :.3f}, cycle_len: {self.gait_cycle_length :.0f}")
    #     # print(f"theta_c_vel: {theta_c_vel :.3f}, gait_state: {self.gait_state :.0f}, swing_prog: {self.swing_progress :.0f}, phase: {0.000 :.3f}, cycle_len: {self.gait_cycle_length :.0f}")

    #     return tau

            

    # def qp_stand(self):
        
    #     if not self.feet_contact: return np.zeros(self.m.nu)
        
    #     alpha = 0.001
    #     Q = np.array([
    #         [1,    0,  0.1],
    #         [0,    1,    0],
    #         [0.1,  0,    5],
    #     ])
        
    #     xc = self.d.qpos[0]
    #     yc = self.d.qpos[2]
    #     theta_c_quat = self.d.qpos[3:7]
    #     theta_c = R.from_quat(theta_c_quat, scalar_first=True).as_euler('zyx')[1]
        
    #     xc_dot = self.d.qvel[0]
    #     yc_dot = self.d.qvel[2]
    #     theta_c_dot = self.d.qvel[4]
        
    #     xc_des = 0
    #     yc_des = 0.5
    #     theta_c_des = 0 
    #     xc_dot_des = 0
    #     yc_dot_des = 0
    #     theta_c_dot_des = 0
               
    #     Kp_x = 5
    #     Kd_x = 10
    #     Kp_y = 5
    #     Kd_y = 10
    #     Kp_theta = 0.5 
    #     Kd_theta = 0.5 

    #     x_ddot_des = Kp_x * (xc_des - xc) + Kd_x * (xc_dot_des - xc_dot)
    #     y_ddot_des = Kp_y * (yc_des - yc) + Kd_y * (yc_dot_des - yc_dot)
    #     theta_ddot_des = Kp_theta * (theta_c_des - theta_c) + Kd_theta * (theta_c_dot_des - theta_c_dot)
        
    #     PF1, PF2 = get_feet_xpos(self.m, self.d)[:, :]
    #     PF1x, PF1y = PF1[[0, 2]]
    #     PF2x, PF2y = PF2[[0, 2]]
        
    #     A = np.array([
    #         [1,       0,       1,       0      ],
    #         [0,       1,       0,       1      ],
    #         [yc-PF1y, PF1x-xc, yc-PF2y, PF2x-xc]
    #     ])
        
    #     b = np.array([
    #         [self.M * x_ddot_des],
    #         [self.M * (y_ddot_des + self.g)],
    #         [self.Izz * theta_ddot_des],
    #     ])
        
    #     n = A.shape[1]
        
    #     Fy_max = 250
    #     Fy_min = 10
    #     mu = 0.7

    #     P = 2 * (A.T @ Q @ A + alpha * np.eye(n))
    #     q = -2 * A.T @ Q @ b
    #     G = np.array([
    #         [0,    1,   0,   0],
    #         [0,    0,   0,   1],
    #         [0,   -1,   0,   0],
    #         [0,    0,   0,  -1],
    #         [1,  -mu,   0,   0],
    #         [0,    0,   1, -mu],
    #         [-1, -mu,   0,   0],
    #         [0,    0,  -1, -mu]
    #     ])
    #     h = np.array([
    #         [Fy_max],
    #         [Fy_max],
    #         [-Fy_min],
    #         [-Fy_min],
    #         [0],
    #         [0],
    #         [0],
    #         [0]
    #     ])
        
    #     P_cvx = matrix(P.astype(float))
    #     q_cvx = matrix(q.astype(float))
    #     G_cvx = matrix(G.astype(float))
    #     h_cvx = matrix(h.astype(float))
        
    #     sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    #     F_GRF_star = np.array([sol['x']])[0]
        
    #     F_feet = -F_GRF_star
        
    #     F_foot_l = np.vstack([ F_feet[0], 0, F_feet[1] ])
    #     F_foot_r = np.vstack([ F_feet[2], 0, F_feet[3] ])

    #     jacp_l = np.zeros((3, self.m.nv))
    #     mujoco.mj_jacGeom(self.m, self.d, jacp_l, None, self.l_foot_id)

    #     jacp_r = np.zeros((3, self.m.nv))
    #     mujoco.mj_jacGeom(self.m, self.d, jacp_r, None, self.r_foot_id)
        
    #     tau_l_full = jacp_l.T @ F_foot_l
    #     tau_r_full = jacp_r.T @ F_foot_r
        
    #     tau_l = tau_l_full[-4:]
    #     tau_r = tau_r_full[-4:]
        
    #     tau = (tau_l + tau_r).flatten()
        
    #     return tau
    

    def mpc(self, x0, xf):
        
        if not self.feet_contact: return np.zeros(self.m.nu)
                
        nv_c = 6
        xc, yc = x0[:2]
        
        PF1, PF2 = get_feet_xpos(self.m, self.d)[:, :]
        PF1x, PF1y = PF1[[0, 2]]
        PF2x, PF2y = PF2[[0, 2]]
        
        A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        B = np.array([
            [0,                                           0,                      0,                      0],
            [0,                                           0,                      0,                      0],
            [0,                                           0,                      0,                      0],
            [1/self.M,                                    0,               1/self.M,                      0],
            [0,                                    1/self.M,                      0,               1/self.M],
            [(1/self.Izz)*(yc-PF1y), (1/self.Izz)*(PF1x-xc), (1/self.Izz)*(yc-PF2y), (1/self.Izz)*(PF2x-xc)]
        ])    
        w = np.array([
            [0],
            [0],
            [0],
            [0],
            [-self.g],
            [0]
        ])

        Q = np.array([
            [100,      0,      1,      0,      0,      0],   # x position
            [0,       10,      0,      0,      0,      0],   # y position
            [1,        0,   100,      0,      0,      0],   # theta - CRITICAL
            [0,        0,      0,     10,      0,      0],   # x_dot
            [0,        0,      0,      0,     10,      0],   # y_dot
            [0,        0,      0,      0,      0,    100],   # theta_dot
        ])
        R = 0.1 * np.eye(4)
        H = 2 * Q  # Terminal cost
        
        # Setup optimization problem
        N = 10
        nx = A.shape[0]
        nm = B.shape[1]
        
        # Discretize continuous-time dynamics
        Bbar = np.hstack([B, w])
        Ak, Bbark = discretize(A, Bbar, self.dt)
        Bk = Bbark[:, :4]
        wk = Bbark[:, 4]

        x0 = np.array(x0).reshape(-1, 1)
        Xf = np.tile(xf, reps=N).reshape(-1, 1)
        W = np.tile(wk, reps=N).reshape(-1, 1)

        # Build prediction matrices
        Phi = np.zeros(shape=(N*nx, nx))
        Gamma = np.zeros(shape=(N*nx, N*nm))
        Omega = np.zeros(shape=(N*nx, N*nx))
        
        for i in range(N):
            Phi[i*nx:(i+1)*nx, :] = np.linalg.matrix_power(Ak, i+1)
            for j in range(i+1):
                Gamma[i*nx:(i+1)*nx, j*nm:(j+1)*nm] = np.linalg.matrix_power(Ak, i-j) @ Bk
                Omega[i*nx:(i+1)*nx, j*nx:(j+1)*nx] = np.linalg.matrix_power(Ak, i-j)
                
        # Build block diagonal cost matrices
        Qbar = np.kron(np.eye(N), Q)
        Qbar[(N-1) * nv_c:, (N-1) * nv_c:] = H
        Rbar = np.kron(np.eye(N), R)
        
        # Contact constraints
        P = 2 * (Gamma.T @ Qbar @ Gamma + Rbar)
        q = 2 * Gamma.T @ Qbar @ (Phi @ x0 + Omega @ W - Xf)
        
        Gu = np.array([
            [0,         1,   0,        0],
            [0,         0,   0,        1],
            [0,        -1,   0,        0],
            [0,         0,   0,       -1],
            [1,  -self.mu,   0,        0],
            [0,         0,   1, -self.mu],
            [-1, -self.mu,   0,        0],
            [0,         0,  -1, -self.mu]
        ])
        hu = np.array([
            [self.Fy_max],
            [self.Fy_max],
            [-self.Fy_min],
            [-self.Fy_min],
            [0],
            [0],
            [0],
            [0]
        ])
        
        G = np.kron(np.eye(N), Gu)
        h = np.tile(hu, (N, 1))

        P_cvx = matrix(P.astype(float))
        q_cvx = matrix(q.astype(float))
        G_cvx = matrix(G.astype(float))
        h_cvx = matrix(h.astype(float))
        
        sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
        F_GRF_star = np.array([sol['x']])[0]
        
        F_feet = -F_GRF_star
        
        F_foot_l = np.vstack([ F_feet[0], 0, F_feet[1] ])
        F_foot_r = np.vstack([ F_feet[2], 0, F_feet[3] ])

        jacp_l = np.zeros((3, self.m.nv))
        mujoco.mj_jacGeom(self.m, self.d, jacp_l, None, self.l_foot_id)

        jacp_r = np.zeros((3, self.m.nv))
        mujoco.mj_jacGeom(self.m, self.d, jacp_r, None, self.r_foot_id)
        
        tau_l_full = jacp_l.T @ F_foot_l
        tau_r_full = jacp_r.T @ F_foot_r
        
        tau_l = tau_l_full[-4:]
        tau_r = tau_r_full[-4:]
        
        tau = (tau_l + tau_r).flatten()
        
        return tau
        


    # def mpc(self, x0, xf):
    #     pass
    #     """
    #     Model predictive controller on centroidal state.

    #     x0: current centroidal state, shape (6,) or (6,1)
    #     xf: desired centroidal state, shape (6,) or (6,1)
    #     """

    #     # Quick exit if not in double support
    #     if not self.feet_contact:
    #         return np.zeros(self.m.nu)

    #     # Debug controls (safe even if attributes do not exist)
    #     debug = True
    #     log_path = "mpc_debug.log"

    #     # Centroid position
    #     x0 = np.asarray(x0).reshape(-1, 1)
    #     xf = np.asarray(xf).reshape(-1, 1)
    #     xc, yc = float(x0[0]), float(x0[1])

    #     # Foot positions in world frame (x, z used)
    #     PF1, PF2 = get_feet_xpos(self.m, self.d)[:, :]
    #     PF1x, PF1y = PF1[[0, 2]]
    #     PF2x, PF2y = PF2[[0, 2]]

    #     # Continuous-time centroidal dynamics
    #     A = np.array([
    #         [0, 0, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 1, 0],
    #         [0, 0, 0, 0, 0, 1],
    #         [0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0]
    #     ])

    #     B = np.array([
    #         [0,                                              0,                       0,                       0],
    #         [0,                                              0,                       0,                       0],
    #         [0,                                              0,                       0,                       0],
    #         [1.0 / self.M,                                  0,               1.0 / self.M,                   0],
    #         [0,                                      1.0 / self.M,                   0,               1.0 / self.M],
    #         [(1.0 / self.Izz) * (yc - PF1y), (1.0 / self.Izz) * (PF1x - xc),
    #          (1.0 / self.Izz) * (yc - PF2y), (1.0 / self.Izz) * (PF2x - xc)]
    #     ])

    #     # Gravity term
    #     w = np.array([
    #         [0.0],
    #         [0.0],
    #         [0.0],
    #         [0.0],
    #         [-self.g],
    #         [0.0],
    #     ])

    #     # ------------------------------------------------------------------
    #     # COST TUNING (initial suggestion)
    #     #
    #     # State: [x, y, theta, x_dot, y_dot, theta_dot]
    #     # Position and orientation more heavily weighted than velocities.
    #     # ------------------------------------------------------------------
    #     Q = np.diag([
    #         150.0,   # x
    #         50.0,    # y
    #         800.0,   # theta (upright)
    #         30.0,    # x_dot
    #         60.0,    # y_dot
    #         80.0     # theta_dot
    #     ])

    #     R = 0.05 * np.eye(4)  # penalize large GRFs a bit more than before
    #     H = 2.0 * Q           # terminal cost, can be increased later

    #     # MPC horizon
    #     N = 10
    #     nx = A.shape[0]
    #     nm = B.shape[1]

    #     # Discretize continuous-time dynamics with zero-order hold
    #     Bbar = np.hstack([B, w])
    #     Ak, Bbark = discretize(A, Bbar, self.dt)
    #     Bk = Bbark[:, :nm]
    #     wk = Bbark[:, nm]

    #     # Build reference and disturbance stacks
    #     Xf = np.tile(xf, reps=(N, 1))
    #     W = np.tile(wk.reshape(-1, 1), reps=(N, 1))

    #     # Build prediction matrices
    #     Phi = np.zeros((N * nx, nx))
    #     Gamma = np.zeros((N * nx, N * nm))
    #     Omega = np.zeros((N * nx, N * nx))

    #     for i in range(N):
    #         Ak_i = np.linalg.matrix_power(Ak, i + 1)
    #         Phi[i * nx:(i + 1) * nx, :] = Ak_i

    #         for j in range(i + 1):
    #             Ak_ij = np.linalg.matrix_power(Ak, i - j)
    #             Gamma[i * nx:(i + 1) * nx, j * nm:(j + 1) * nm] = Ak_ij @ Bk
    #             Omega[i * nx:(i + 1) * nx, j * nx:(j + 1) * nx] = Ak_ij

    #     # Block-diagonal cost matrices
    #     Qbar = np.kron(np.eye(N), Q)
    #     Qbar[(N - 1) * nx:, (N - 1) * nx:] = H
    #     Rbar = np.kron(np.eye(N), R)

    #     # Contact constraints, use class parameters
    #     # Fy >= Fy_min, Fy <= Fy_max, |Fx| <= mu * Fy
    #     Gu = np.array([
    #         [0,           1,   0,          0],
    #         [0,           0,   0,          1],
    #         [0,          -1,   0,          0],
    #         [0,           0,   0,         -1],
    #         [1,   -self.mu,   0,          0],
    #         [0,           0,   1,   -self.mu],
    #         [-1, -self.mu,   0,          0],
    #         [0,           0,  -1,   -self.mu]
    #     ])

    #     hu = np.array([
    #         [self.Fy_max],
    #         [self.Fy_max],
    #         [-self.Fy_min],
    #         [-self.Fy_min],
    #         [0.0],
    #         [0.0],
    #         [0.0],
    #         [0.0],
    #     ])

    #     G = np.kron(np.eye(N), Gu)
    #     h = np.tile(hu, (N, 1))

    #     # QP matrices
    #     P = 2.0 * (Gamma.T @ Qbar @ Gamma + Rbar)
    #     q_vec = 2.0 * Gamma.T @ Qbar @ (Phi @ x0 + Omega @ W - Xf)

    #     P_cvx = matrix(P.astype(float))
    #     q_cvx = matrix(q_vec.astype(float))
    #     G_cvx = matrix(G.astype(float))
    #     h_cvx = matrix(h.astype(float))

    #     # Solve QP with basic robustness
    #     try:
    #         sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    #     except Exception as e:
    #         if debug:
    #             print(f"[MPC] QP exception at t={self.d.time:.4f}: {e}")
    #             try:
    #                 with open(log_path, "a") as f:
    #                     f.write(f"t={self.d.time:.4f}, status=exception, error={repr(e)}\n")
    #             except Exception:
    #                 pass
    #         return np.zeros(self.m.nu)

    #     status = sol.get("status", "unknown")

    #     if status != "optimal":
    #         if debug:
    #             print(f"[MPC] QP status not optimal at t={self.d.time:.4f}: {status}")
    #             try:
    #                 with open(log_path, "a") as f:
    #                     f.write(f"t={self.d.time:.4f}, status={status}\n")
    #             except Exception:
    #                 pass
    #         return np.zeros(self.m.nu)

    #     # Full input sequence as column vector, shape (N*nm, 1)
    #     U_star = np.array(sol["x"]).reshape(-1, 1)

    #     # Original layout used for ground reaction forces
    #     F_GRF_star = np.array([sol["x"]])[0]  # keep your original convention
    #     F_feet = -F_GRF_star  # you stated explicitly that this sign is correct

    #     # First-step GRFs, for logging and sanity check
    #     u0 = U_star[:nm].flatten()

    #     # Predicted state trajectory for debugging
    #     X_pred = Phi @ x0 + Gamma @ U_star + Omega @ W
    #     xN_pred = X_pred[(N - 1) * nx:N * nx, :]
    #     xN_err = xN_pred - xf

    #     # Constraint slack for first step, useful to see friction saturation
    #     gu_u0 = Gu @ u0.reshape(-1, 1)
    #     slack0 = h[:Gu.shape[0], :] - gu_u0
    #     min_slack0 = float(np.min(slack0))

    #     # Map GRFs to joint torques
    #     F_foot_l = np.vstack([F_feet[0], 0.0, F_feet[1]])
    #     F_foot_r = np.vstack([F_feet[2], 0.0, F_feet[3]])

    #     jacp_l = np.zeros((3, self.m.nv))
    #     mujoco.mj_jacGeom(self.m, self.d, jacp_l, None, self.l_foot_id)

    #     jacp_r = np.zeros((3, self.m.nv))
    #     mujoco.mj_jacGeom(self.m, self.d, jacp_r, None, self.r_foot_id)

    #     tau_l_full = jacp_l.T @ F_foot_l
    #     tau_r_full = jacp_r.T @ F_foot_r

    #     tau_l = tau_l_full[-4:]
    #     tau_r = tau_r_full[-4:]

    #     tau = (tau_l + tau_r).flatten()

    #     # Debug printing and logging
    #     if debug:
    #         t_sim = float(self.d.time)
    #         x_err_now = (x0 - xf).flatten()
    #         tau_norm = float(np.linalg.norm(tau))
    #         u0_norm = float(np.linalg.norm(u0))
    #         obj = float(sol["primal objective"])

    #         print(
    #             f"[MPC] t={t_sim:.3f}, "
    #             f"status={status}, "
    #             f"obj={obj:.3e}, "
    #             f"||x0-xf||={np.linalg.norm(x_err_now):.3e}, "
    #             f"||xN-xf||={np.linalg.norm(xN_err):.3e}, "
    #             f"||u0||={u0_norm:.3e}, "
    #             f"||tau||={tau_norm:.3e}, "
    #             f"min_slack0={min_slack0:.3e}"
    #         )

    #         try:
    #             with open(log_path, "a") as f:
    #                 f.write(
    #                     "t={:.4f},status={},obj={:.6e},"
    #                     "x0={},xf={},xN_err={},"
    #                     "u0={},tau={},min_slack0={:.6e}\n".format(
    #                         t_sim,
    #                         status,
    #                         obj,
    #                         x0.flatten().tolist(),
    #                         xf.flatten().tolist(),
    #                         xN_err.flatten().tolist(),
    #                         u0.tolist(),
    #                         tau.tolist(),
    #                         min_slack0,
    #                     )
    #                 )
    #         except Exception:
    #             # File logging problems should never crash the controller
    #             pass

    #     return tau


    
