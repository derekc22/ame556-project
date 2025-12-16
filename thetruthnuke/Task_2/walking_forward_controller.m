function [cntl, foot_out] = walking_forward_controller(full_state, t)
    global params

    % Get the foot positions from forward kinematics
    FK = robot_FK(full_state(1:3), full_state(4:7), params.l, params.a);
    P_foot = [FK.p2; FK.p4]; % [x_p1; y_p1; x_p2; y_p2]
    foot_out(1:4) = P_foot'; % store the feet positions
    
    % calculate foot velocities
    dP_foot = get_foot_velocities(full_state);
    foot_out(5:8) = dP_foot'; %store the feet velocities
    
    % Extract individual foot positions and velocities
    P1 = P_foot(1:2);
    P2 = P_foot(3:4);
    dP1 = dP_foot(1:2);
    dP2 = dP_foot(3:4);
    
    % Get gait schedule
    con = gait_schedule(params.N, params.dt, t);
    
    if con(1,1) && con(2,1)
        % Double Contact
        p_d = p_body;
        F = mpc(full_state, t, P_foot, con, p_d);
        cntl = force_torque_map(full_state, F);
        
    elseif con(1,1) && ~con(2,1)
        %Left Foot Stance, Right foot Swing
        [F_swing, p_d, foot_desired] = swing_leg_control(full_state, P2, dP2, P1, params.N, params.dt, t);
        F = mpc(full_state, t, P_foot, con, p_d);
        F_new = [F(1:2); -F_swing];
        cntl = force_torque_map(full_state, F_new);
        
    elseif con(2,1) && ~con(1,1)
        % Right Foot Stance, Left Foot Swing
        [F_swing, p_d, foot_desired] = swing_leg_control(full_state, P1, dP1, P2, params.N, params.dt, t);
        F = mpc(full_state, t, P_foot, con, p_d);
        F_new = [-F_swing; F(3:4)];
        cntl = force_torque_map(full_state, F_new);
        
    else
        % Flight Phase
        cntl = zeros(4,1);
        foot_desired = zeros(1,4);
    end
    foot_out(9:12) = foot_desired;
end

%% Helper Functions

function F = mpc(full_state, t, P_foot, con, p_d)
    global params
    
    % 1. Build current continuous dynamics matrices A and B
    [A, B] = build_curr_dynamics(full_state, P_foot);

    % 2. Discretize dynamics matrices
    [A_k, B_k] = discretize_dynamics(A, B, params.dt, true);
    
    % 3. Determine dimensions
    params.num_states = size(A, 1);
    params.num_inputs = size(B, 2);
    
    % 4. Extract SRBD state
    X_srbd = get_SRBD_state(full_state);
    
    % 5. Get reference trajectory
    X_ref = get_reference_trajectory(params.dt, params.N, X_srbd, p_d);
    y = reshape(X_ref, [params.num_states*params.N, 1]);
    
    % 6. Build condensed matrices Aqp and Bqp
    [Aqp, Bqp] = build_condensed_matrices(A_k, B_k, params.N);
    
    % 7. Build H and f matrices using condensed formulation
    L = kron(eye(params.N), params.Q);
    K = kron(eye(params.N), params.R);
    
    H = 2*(Bqp'*L*Bqp + K);
    f = 2*Bqp'*L*(Aqp*X_srbd - y);
    
    % Ensure H is symmetric
    H = (H + H')/2;

    % 8. Make inequality constraints with contact schedule
    [A_ineq, B_ineq] = make_inequality_constraints_walking(params.N, con);
    
    % 9. Call quadprog
    options = optimoptions('quadprog', ...
                           'Algorithm', 'interior-point-convex', ...
                           'MaxIterations', 1000, ...
                           'TolFun', 1e-6, ...
                           'TolX', 1e-8, ...
                           'Display', 'off');
    
    u_mpc = quadprog(H, f, A_ineq, B_ineq, [], [], [], [], [], options);

    % 10. Extract first control input (forces)
    F = u_mpc(1:params.num_inputs);
end

function X_ref = get_reference_trajectory(dt, N, X_current, p_d)
    global params
        
    X_d = [p_d(1);           % desired x position
           0.45;             % desired y position
           0;                % desired theta
           params.dx_des;    % desired dx
           0;                % desired dy
           0;                % desired dtheta
           params.g];        % gravity
    
    % Initialize reference with current state
    X_ref = repmat(X_current, 1, N);
    
    % Build reference trajectory by integrating desired velocities
    for i = 1:3  % For x, y, theta
        for k = 2:N
            if X_d(i+3) ~= 0
                % Non-zero desired velocity: integrate forward
                X_ref(i, k) = X_current(i) + X_d(i+3)*((k-1)*dt);
            else
                % Zero desired velocity: use desired position
                X_ref(i, k) = X_d(i);
            end
        end
    end
    
    % Set desired velocities for all timesteps
    X_ref(4:6, :) = repmat(X_d(4:6), 1, N);
    
    % Set gravity for all timesteps
    X_ref(7, :) = params.g;
end

function [F_swing, p_CG_des, foot_desired] = swing_leg_control(full_state, P_swing, dP_swing, P_support, N, dt, t)
    % PD control for swing leg trajectory
    global params
    
    % Extract body state for foot placement
    p = full_state(1:2);    % [x; y]
    dp = full_state(8:9);   % [dx; dy]
    
    % Time in swing phase
    ts = rem(t, N*dt/2);
    
    % X-position desired (Raibert Heuristic)
    delT = N*dt/2;
    Px_d = p(1) + (1/2)*delT*params.dx_des - params.Kv*(params.dx_des - dp(1));
    Px_dot_d = 0; % Desired velocity is 0 at the target foot placement
    
    % Y-position desired (sinusoidal)
    Py_d = params.swing_height * sin(pi*ts/delT);
    Py_dot_d = (params.swing_height * pi/delT) * cos(pi*ts/delT);

    % Desired COM position
    px_d = (P_swing(1) + P_support(1))/2;
    p_CG_des = [px_d; params.ypos_des];
    
    % PD control law
    F_swing = params.Kp*([Px_d; Py_d] - P_swing) + params.Kd*([Px_dot_d; Py_dot_d] - dP_swing);

    %store desired for plotting
    foot_desired = [Px_d, Py_d, Px_dot_d, Py_dot_d];
end

function sigma = gait_schedule(N, dt, t)
    % Generate gait schedule based on walking phase
    % Returns contact schedule: sigma(1,:) = left foot, sigma(2,:) = right foot
    
    phase = floor(t/dt);
    k = rem(phase, N);
    
    % Base pattern: alternating contact
    sigma_l0 = [ones(1, N/2) zeros(1, N/2)];
    sigma_r0 = [zeros(1, N/2) ones(1, N/2)];
    
    % Shift pattern based on current phase
    if k ~= 0
        sigma_l = [sigma_l0(k+1:N) sigma_l0(1:k)];
        sigma_r = [sigma_r0(k+1:N) sigma_r0(1:k)];
    else
        sigma_l = sigma_l0;
        sigma_r = sigma_r0;
    end
    
    sigma = [sigma_l; sigma_r];
end

function dP_foot = get_foot_velocities(full_state)
    % Compute foot velocities in the WORLD FRAME
    
    % 1. Extract states
    dx = full_state(8);   % CoM linear velocity (World X)
    dy = full_state(9);   % CoM linear velocity (World Y)
    theta = full_state(3);
    q = full_state(4:7);
    dtheta = full_state(10);
    dq = full_state(11:14);
    
    % 2. Get Jacobian matrix (for relative velocities)
    J = foot_vel_jacobian(theta, q);
    
    % 3. Calculate Relative Foot Velocities
    % dP_foot_rel = J * [dtheta; dq] 
    dP_foot_rel = J * [dtheta; dq];
    
    % 4. Calculate Absolute Foot Velocities (World Frame)
    % P_dot_World = P_dot_CoM_Linear + P_dot_Relative
    dP_foot = dP_foot_rel + [dx; dy; dx; dy];
end

function J = foot_vel_jacobian(theta, q)
    % Compute Jacobian for foot velocities
    global params
    l = params.l;
    
    q1 = q(1);
    q2 = q(2);
    q3 = q(3);
    q4 = q(4);
    
    % Left foot Jacobian (foot 1) 
   
    J11 = -l*sin(q1 + theta) - l*sin(q1 + q2 + theta); 
    J12 = l*cos(q1 + theta) + l*cos(q1 + q2 + theta);  
    
    J13 = -l*sin(q1 + q2 + theta);                    
    J14 = l*cos(q1 + q2 + theta);                     
    
    % Right foot Jacobian (foot 2) 
    
    J21 = -l*sin(q3 + theta) - l*sin(q3 + q4 + theta); 
    J22 = l*cos(q3 + theta) + l*cos(q3 + q4 + theta);  
    
    J23 = -l*sin(q3 + q4 + theta);                    
    J24 = l*cos(q3 + q4 + theta);                     
    
    % Full Jacobian: [x1_dot; y1_dot; x2_dot; y2_dot] = J * [theta_dot; q1_dot; q2_dot; q3_dot; q4_dot]
    J = [J11, J11, J13, 0, 0;
         J12, J12, J14, 0, 0;
         J21, 0, 0, J21, J23;
         J22, 0, 0, J22, J24];
end

function [A_ineq, B_ineq] = make_inequality_constraints_walking(N, con)
    % Build inequality constraints with contact-aware force limits
    global params

    m = params.num_inputs; 
    mu = params.mu;

    % Friction cone constraints (4 per timestep) + force limits (4 per timestep)
    n_friction = 4;
    n_force = 4;
    n_rows = n_friction + n_force;
    
    A_ineq = zeros(N*n_rows, N*m);
    B_ineq = zeros(N*n_rows, 1);

    for i = 1:N
        ui_idx = (i-1)*m + 1 : i*m;
        friction_idx = (i-1)*n_rows + 1 : (i-1)*n_rows + n_friction;
        force_idx = (i-1)*n_rows + n_friction + 1 : i*n_rows;

        % Friction cone block
        friction_block = zeros(n_friction, m);
        
        % Foot 1 friction: |Fx1| <= μ*Fy1
        friction_block(1,1) =  1;
        friction_block(1,2) = -mu;
        friction_block(2,1) = -1;
        friction_block(2,2) = -mu;
        
        % Foot 2 friction: |Fx2| <= μ*Fy2
        friction_block(3,3) =  1;
        friction_block(3,4) = -mu;
        friction_block(4,3) = -1;
        friction_block(4,4) = -mu;
        
        A_ineq(friction_idx, ui_idx) = friction_block;
        
        % Force limit block (contact-aware)
        force_block = zeros(n_force, m);
        force_block(1, 2) = 1;  % Fy1 <= max
        force_block(2, 2) = -1; % Fy1 >= min
        force_block(3, 4) = 1;  % Fy2 <= max
        force_block(4, 4) = -1; % Fy2 >= min
        
        A_ineq(force_idx, ui_idx) = force_block;
        
        % B_ineq: force limits based on contact schedule
        params.Fy_constraint = [-10 250];
        B_ineq(force_idx) = [params.Fy_constraint(2)*con(1,i); 
                            params.Fy_constraint(1)*con(1,i); 
                            params.Fy_constraint(2)*con(2,i); 
                            params.Fy_constraint(1)*con(2,i)];
    end
end

function [Aqp, Bqp] = build_condensed_matrices(Ak, Bk, N)
    % Build the condensed prediction matrices
    % Aqp: stacks powers of Ak
    % Bqp: lower triangular block structure
    
    n = size(Ak, 1);
    m = size(Bk, 2);
    
    % Build Aqp
    Aqp = zeros(N*n, n);
    for i = 1:N
        Aqp((i-1)*n+1:i*n, :) = Ak^i;
    end
    
    % Build Bqp
    Bqp = zeros(N*n, N*m);
    for i = 0:N-1
        element = Ak^i * Bk;
        % Place element in the appropriate diagonal blocks
        for j = 1:N-i
            row_idx = (i+j-1)*n+1:(i+j)*n;
            col_idx = (j-1)*m+1:j*m;
            Bqp(row_idx, col_idx) = element;
        end
    end
end

function [A, B] = build_curr_dynamics(full_state, P_foot)
    % Build the continuous A and B matrices for the SRBD
    % SRBD state x = [x, y, theta, dx, dy, dtheta, g] 
    global params

    A = [0, 0, 0, 1, 0, 0, 0;
         0, 0, 0, 0, 1, 0, 0;
         0, 0, 0, 0, 0, 1, 0;
         0, 0, 0, 0, 0, 0, 0;
         0, 0, 0, 0, 0, 0, -1;
         0, 0, 0, 0, 0, 0, 0;
         0, 0, 0, 0, 0, 0, 0];
    
    % Get r vectors from foot to CG
    [r1, r2] = get_rvec(full_state, P_foot);

    B = [0, 0, 0, 0;
         0, 0, 0, 0;
         0, 0, 0, 0;
         (1/params.m), 0, (1/params.m), 0;
         0, (1/params.m), 0, (1/params.m);
         (-r1(2)/params.I), (r1(1)/params.I), (-r2(2)/params.I), (r2(1)/params.I);
         0, 0, 0, 0];
end

function [Ad, Bd] = discretize_dynamics(A, B, dt, continuous)
    n = size(A, 1);
    
    if continuous
        % Euler discretization
        Ad = eye(n) + A * dt;
        Bd = B * dt;
    else
        Ad = A;
        Bd = B;
    end
end

function [r1, r2] = get_rvec(full_state, Pfoot)
    % Get the r vectors from foot positions to body CG
    Pcg = full_state(1:2); % [x; y]

    % Left foot
    r1 = Pfoot(1:2) - Pcg;

    % Right foot
    r2 = Pfoot(3:4) - Pcg;
end

function torques = force_torque_map(full_state, u)
    % Map ground reaction forces to joint torques via Jacobian transpose
    global params
    l = params.l;
    theta = full_state(3);
    q1 = full_state(4);
    q2 = full_state(5);
    q3 = full_state(6);
    q4 = full_state(7);

    J11 = [l*cos(q1 + theta) + l*cos(q1+q2+theta), l*cos(q1 + q2 + theta);
           l*sin(q1 + theta) + l*sin(q1+q2+theta), l*sin(q1 + q2 + theta)];
    
    J22 = [l*cos(q3 + theta) + l*cos(q3 + q4 + theta), l*cos(q3 + q4 + theta);
           l*sin(q3 + theta) + l*sin(q3 + q4 + theta), l*sin(q3 + q4 + theta)];

    % Build full Jacobian
    J = [J11, zeros(2);
         zeros(2), J22];

    % tau = -J'*F
    torques = -J' * u;
end

function X_srbd = get_SRBD_state(full_state)
    global params
    % Extract SRBD state from full simulation state
    % full state X = [x, y, theta, q1, q2, q3, q4, x_dot, y_dot, theta_dot,
    %                 q1_dot, q2_dot, q3_dot, q4_dot]
    % SRBD state X_srbd = [x, y, theta, x_dot, y_dot, theta_dot, g]

    X_srbd = [full_state(1:3);
              full_state(8:10);
              params.g];
end