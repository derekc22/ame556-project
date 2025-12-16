function cntl = mpc_controller_task_two(full_state, t)
    global params

    % get the foot positions from forward kinematics
    FK = robot_FK(full_state(1:3), full_state(4:7), params.l, params.a);
    P_foot = [FK.p2 ; FK.p4]; %[x_p1; y_p1 ; x_p2 ; y_p2;]

    ground_tol = 1e-2;
    foot_contact = abs(P_foot([2,4])) < ground_tol; % y-coordinates near zero

    if all(foot_contact)

        % 1. build current continuous dynamics matrices A and B
        [A, B] = build_curr_dynamics(full_state, P_foot);

        % 2. discretize dynamics matrices
        [A_k, B_k] = discretize_dynamics(A, B, params.dt, true);
        
        % 3. determine dimensions
        params.num_states = size(A, 1);  % number of states
        params.num_inputs = size(B, 2);  % number of inputs
        
        % 4. Extract SRBD state
        X_srbd = get_SRBD_state(full_state);
        
        % 5. Get reference trajectory (now time-aware)
        X_ref = get_reference_trajectory(params.N, params.dt, X_srbd, t);
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

        % 8. make inequality constraints
        [A_ineq, B_ineq] = make_inequality_constraints(params.N);
        
        % 9. call quadprog
        options = optimoptions('quadprog', ...
                               'Algorithm', 'interior-point-convex', ...
                               'MaxIterations', 1000, ...
                               'TolFun',1e-6, ...
                               'TolX',1e-8, ...
                               'Display', 'off');
        
        % No equality constraints needed with condensed formulation
        u_mpc = quadprog(H, f, A_ineq, B_ineq, [], [], [], [], [], options);

        % 10. extract first control input (forces)
        u = u_mpc(1:params.num_inputs);

        % 11. jacobian to determine torques (ctrl)
        cntl = force_torque_map(full_state, u);
    else
        % if no contact, zero force
        cntl = zeros(4,1);
    end
end

%% Helper Functions

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

function X_ref = get_reference_trajectory(N, dt, X_current, t_current)
    % Generate reference trajectory over prediction horizon
    % Now uses interpolated trajectory from waypoints
    global params
    
    numStates = 7; % [x, y, theta, dx, dy, dtheta, g]
    X_ref = zeros(numStates, N);
    
    for k = 1:N
        t_future = t_current + (k-1)*dt;
        
        % Get interpolated desired state at this future time
        x_des_k = get_desired_state(t_future);
        
        % Build full state vector with gravity
        X_ref(:, k) = [x_des_k; params.g];
    end
end

function [A_ineq, B_ineq] = make_inequality_constraints(N)
    global params

    m = params.num_inputs; 
    mu = params.mu;

    % Friction cone constraints: 4 constraints per time step
    nRows = 4;
    A_ineq = zeros(N*nRows, N*m);
    B_ineq = zeros(N*nRows, 1);

    for i = 1:N
        ui_idx = (i-1)*m + 1 : i*m;
        row_idx = (i-1)*nRows + 1 : i*nRows;

        block = zeros(nRows, m);

        % Foot 1 friction constraints: |Fx1| <= μ*Fy1
        block(1,1) =  1;  % Fx1
        block(1,2) = -mu; % -μ*Fy1

        block(2,1) = -1;  % -Fx1
        block(2,2) = -mu; % -μ*Fy1

        % Foot 2 friction constraints: |Fx2| <= μ*Fy2
        block(3,3) =  1;  % Fx2
        block(3,4) = -mu; % -μ*Fy2

        block(4,3) = -1;  % -Fx2
        block(4,4) = -mu; % -μ*Fy2

        A_ineq(row_idx, ui_idx) = block;
        % B_ineq stays zero (constraints are <= 0)
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