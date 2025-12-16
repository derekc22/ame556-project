function [cntl, violation] = enforce_constraints(cntl, Z, saturate_cntl)
    % Joint states
    q  = Z(4:7);
    dq = Z(11:14);

    % Define limits
    global limits
    limits.angle_min = deg2rad([-120; 0; -120; 0]);
    limits.angle_max = deg2rad([ 30; 160; 30; 160]);
    limits.vel_lim   = [30; 15; 30; 15];
    limits.cntl_lim  = [30; 60; 30; 60];

    joint_names = ["q1","q2","q3","q4"];

    % Saturate torques
    if saturate_cntl
        cntl = max(min(cntl, limits.cntl_lim), -limits.cntl_lim);
    end

    violation = false; %preallocate

    % Check joint angles
    idx = find(q < limits.angle_min | q > limits.angle_max);
    if ~isempty(idx)
        violation = true;
        for j = idx'
            warning('Angle limit violated at %s (q = %.1f°).', joint_names(j), rad2deg(q(j)));
        end
    end

    % Check joint velocities
    idx = find(abs(dq) > limits.vel_lim);
    if ~isempty(idx)
        violation = true;
        for j = idx'
            warning('Velocity limit violated at %s (dq = %.1f rad/s).', joint_names(j), dq(j));
        end
    end

    % Check torques
    idx = find(abs(cntl) > limits.cntl_lim + 1e-9);
    if ~isempty(idx)
        violation = true;
        for j = idx'
            warning('Torque limit exceeded at %s (cntl = %.1f Nm).', joint_names(j), cntl(j));
        end
    end
end

