function [cntl, foot_out] = running_controller(full_state, t)
    global params

    % Get time-varying parameters
    params_t = get_scheduled_params(t);

    % Get the foot positions from forward kinematics
    FK = robot_FK(full_state(1:3), full_state(4:7), params.l, params.a);
    P_foot = [FK.p2; FK.p4]; % [x_p1; y_p1; x_p2; y_p2]
    foot_out(1:4) = P_foot'; % store the feet positions
    
    % calculate foot velocities
    dP_foot = get_foot_velocities(full_state);
    foot_out(5:8) = dP_foot'; %store the feet velocities for plotting
    
    % Extract individual foot positions and velocities
    P1 = P_foot(1:2);
    P2 = P_foot(3:4);
    dP1 = dP_foot(1:2);
    dP2 = dP_foot(3:4);
    
    % Get gait schedule
    contact = gait_schedule(params.N, params.dt, t, params_t);
    foot_desired = zeros(1,4);     % Initialize foot_desired
    
    if contact(1,1) && contact(2,1)
        % Double Contact
        pCG_des = p_body(full_state, t, params_t); % get desired CG position
        F = mpc(full_state, t, P_foot, contact, pCG_des, params_t); % calcualte the required forces
        cntl = force_torque_map(full_state, F); % map to torques directly
        
    elseif contact(1,1) && ~contact(2,1)
        % Left Foot Stance, Right foot Swing
        [F_swing, pCG_des, foot_desired] = swing_leg_control(full_state, P2, dP2, P1, t, params_t); % call swing leg control
        F = mpc(full_state, t, P_foot, contact, pCG_des, params_t);
        F_total = [F(1:2); -F_swing];
        cntl = force_torque_map(full_state, F_total);
        
    elseif contact(2,1) && ~contact(1,1)
        % Right Foot Stance, Left Foot Swing
        [F_swing, pCG_des, foot_desired] = swing_leg_control(full_state, P1, dP1, P2, t, params_t);
        F = mpc(full_state, t, P_foot, contact, pCG_des, params_t);
        F_total = [-F_swing; F(3:4)];
        cntl = force_torque_map(full_state, F_total);
        
    else
        % Flight Phase
        F_swing1 = swing_leg_control_flight(full_state, P1, dP1, t, params_t);
        F_swing2 = swing_leg_control_flight(full_state, P2, dP2, t, params_t);
        F_total = [-F_swing1; -F_swing2];
        cntl = force_torque_map(full_state, F_total);
    end
    foot_out(9:12) = foot_desired;
end

%% Helper Functions

function pCG_des = p_body(full_state, t, params_t)
    % Compute desired body position during double contact
    global params
    p = full_state(1:2);
    
    if isfield(params, 'schedule') && isfield(params.schedule, 'dx_des')
        n_steps = max(100, ceil(t / params.dt));
        t_vec = linspace(0, t, n_steps);
        dx_vec = zeros(size(t_vec));
        
        for i = 1:length(t_vec)
            params_temp = get_scheduled_params(t_vec(i));
            dx_vec(i) = params_temp.dx_des;
        end
        
        x_des = trapz(t_vec, dx_vec);
    else
        % Constant velocity case
        x_des = params_t.dx_des * t;
    end
    
    pCG_des = [x_des; params_t.ypos_des];
end

function F = mpc(full_state, t, P_foot, contact, pCG_des, params_t)
    global params
    
    % Check if no contact, if so return zero
    if ~any(contact(:, 1))
        F = zeros(4, 1);
        return;
    end
    
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
    X_ref = get_reference_trajectory(params.dt, params.N, X_srbd, pCG_des, params_t);
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
    [A_ineq, B_ineq] = make_inequality_constraints_walking(params.N, contact);
    
    % 9. Call quadprog
    options = optimoptions('quadprog', ...
                           'Algorithm', 'interior-point-convex', ...
                           'MaxIterations', 1000, ...
                           'TolFun', 1e-6, ...
                           'TolX', 1e-8, ...
                           'Display', 'off');
    
    u_mpc = quadprog(H, f, A_ineq, B_ineq, [], [], [], [], [], options);

    % 10. Extract first control input
    F = u_mpc(1:params.num_inputs);
end

function X_ref = get_reference_trajectory(dt, N, X_current, pCG_des, params_t)
    global params
        
    X_d = [pCG_des(1);
           pCG_des(2);
           params_t.theta_des;
           params_t.dx_des;
           0;
           0;
           params.g];
    
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
    
    X_ref(4:6, :) = repmat(X_d(4:6), 1, N);
    X_ref(7, :) = params.g;
end

function [F_swing, p_CG_des, foot_desired] = swing_leg_control(full_state, P_swing, dP_swing, P_support, t, params_t)
    % PD control for swing leg trajectory
    global params
    
    p = full_state(1:2);    % [x; y]
    dp = full_state(8:9);   % [dx; dy]
    
    % Use time-based swing phase duration
    gait_period = 2 * (params_t.stance_time + params_t.flight_time);
    swing_duration = params_t.stance_time + params_t.flight_time; % One leg is swinging during stance+flight of other leg
    t_cycle = mod(t, gait_period);
    
    if t_cycle < params_t.stance_time
        % We're in left stance phase - right leg should be swinging
        ts = t_cycle;
    else
        % We're in right stance phase - left leg should be swinging  
        ts = t_cycle - (params_t.stance_time + params_t.flight_time);
        if ts < 0
            ts = 0;
        end
    end
    
    % Normalize time to swing duration
    ts = min(ts, swing_duration);
    
    % X-position desired (Raibert Heuristic)
    Px_d = p(1) + (1/2)*swing_duration*params_t.dx_des - params_t.Kv*(params_t.dx_des - dp(1));
    Px_dot_d = 0; % Desired velocity is 0 at the target foot placement
    
    % Y-position desired (sinusoidal swing trajectory)
    Py_d = params_t.swing_height * sin(pi*ts/swing_duration);
    Py_dot_d = (params_t.swing_height * pi/swing_duration) * cos(pi*ts/swing_duration);

    % Desired COM position
    px_d = (P_swing(1) + P_support(1))/2;
    p_CG_des = [px_d; params_t.ypos_des];
    
    % PD control law
    F_swing = params.Kp*([Px_d; Py_d] - P_swing) + params.Kd*([Px_dot_d; Py_dot_d] - dP_swing);

    %store desired for plotting
    foot_desired = [Px_d, Py_d, Px_dot_d, Py_dot_d];
end

function F_swing = swing_leg_control_flight(full_state, P_swing, dP_swing, t, params_t)
    % PD control for swing leg during flight phase
    global params
    
    % Extract body state
    p = full_state(1:2);
    dp = full_state(8:9);
    
    gait_period = 2 * (params_t.stance_time + params_t.flight_time);
    t_cycle = mod(t, gait_period);
    
    if t_cycle >= params_t.stance_time && t_cycle < (params_t.stance_time + params_t.flight_time)
        % First flight phase (after left stance)
        ts = t_cycle - params_t.stance_time;
    elseif t_cycle >= (params_t.stance_time + params_t.flight_time + params_t.stance_time)
        % Second flight phase (after right stance)
        ts = t_cycle - (params_t.stance_time + params_t.flight_time + params_t.stance_time);
    else
        ts = 0; % error
    end
    
    % X-position desired
    Px_d = p(1) + (1/2)*params_t.flight_time*dp(1);
    Px_dot_d = 0;
    
    % Y-position desired (sinusoidal - continue swing motion during flight)
    Py_d = params_t.swing_height * sin(pi*(params_t.stance_time + ts)/(params_t.stance_time + params_t.flight_time));
    Py_dot_d = (params_t.swing_height * pi/(params_t.stance_time + params_t.flight_time)) * cos(pi*(params_t.stance_time + ts)/(params_t.stance_time + params_t.flight_time));
    
    % PD control law
    F_swing = params.Kp*([Px_d; Py_d] - P_swing) + params.Kd*([Px_dot_d; Py_dot_d] - dP_swing);
end

function sigma = gait_schedule(N, dt, t, params_t)
    % Generate gait schedule with flight phase
    
    gait_period = 2 * (params_t.stance_time + params_t.flight_time);  % Full gait cycle period
    
    sigma = zeros(2, N);
    
    for i = 1:N
        t_pred = t + (i-1)*dt; % t at this prediction step
        t_phase = mod(t_pred, gait_period); % Current phase in gait cycle

        if t_phase < params_t.stance_time
            % Left foot stance
            sigma(1, i) = 1;
            sigma(2, i) = 0;
        elseif t_phase < (params_t.stance_time + params_t.flight_time)
            % Flight phase 1
            sigma(1, i) = 0;
            sigma(2, i) = 0;
        elseif t_phase < (params_t.stance_time + params_t.flight_time + params_t.stance_time)
            % Right foot stance
            sigma(1, i) = 0;
            sigma(2, i) = 1;
        else
            % Flight phase 2
            sigma(1, i) = 0;
            sigma(2, i) = 0;
        end
    end
end

function dP_foot = get_foot_velocities(full_state)
    % Compute foot velocities
    
    % 1. Extract states
    dx = full_state(8);
    dy = full_state(9);
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
    % pCG_desot_World = pCG_desot_CoM_Linear + pCG_desot_Relative
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
        
        % Ensure we don't exceed contact schedule array bounds
        con_idx = min(i, size(con, 2));
        
        B_ineq(force_idx) = [params.Fy_constraint(2)*con(1,con_idx); 
                            params.Fy_constraint(1)*con(1,con_idx); 
                            params.Fy_constraint(2)*con(2,con_idx); 
                            params.Fy_constraint(1)*con(2,con_idx)];
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

    X_srbd = [full_state(1:3);
              full_state(8:10);
              params.g];
end

function params_t = get_scheduled_params(t)
    global params
    
    % Initialize output with current params
    params_t = params;
    
    if ~isfield(params, 'schedule')
        return;
    end
    
    schedule = params.schedule;
    schedulable_params = {'dx_des', 'ypos_des', 'theta_des', ...
                          'stance_time', 'flight_time', 'Kv', 'swing_height'};
    
    % Interpolate each scheduled parameter
    for i = 1:length(schedulable_params)
        param_name = schedulable_params{i};
        
        if isfield(schedule, param_name) && isstruct(schedule.(param_name))
            sched = schedule.(param_name);
            
            % Validate schedule structure
            if ~isfield(sched, 'time') || ~isfield(sched, 'value')
                warning('Schedule for %s missing time or value field', param_name);
                continue;
            end
            
            params_t.(param_name) = interp1(sched.time, sched.value, t, ...
                                            'spline', 'extrap');
            
            % Clamp to endpoint values
            if t < sched.time(1)
                params_t.(param_name) = sched.value(1);
            elseif t > sched.time(end)
                params_t.(param_name) = sched.value(end);
            end
        end
    end
end