function x_des = get_desired_state(t)
    global params
    
    waypoints = params.trajectory_waypoints; % extract waypoints
    t_waypoints = waypoints(:, 1);
    n_cols = size(waypoints, 2);
    
    % Clamp time to trajectory bounds
    t_clamped = max(min(t, t_waypoints(end)), t_waypoints(1));
    
    x_des = zeros(6, 1);
    
    % Check if velocities are provided (7 or 10 columns)
    if n_cols == 4         % Only positions provided -> derive velocities
        dt = 0.001;
        for i = 1:3         % Interpolate positions
            x_des(i) = interp1(t_waypoints, waypoints(:, i+1), t_clamped, params.interpolation_method);
        end
        
        % Compute velocities
        if t_clamped >= t_waypoints(end) - dt/2
            % At end of trajectory - use backward difference
            for i = 1:3
                x_future = x_des(i);
                x_past = interp1(t_waypoints, waypoints(:, i+1), t_clamped - dt, params.interpolation_method);
                x_des(i+3) = (x_future - x_past) / dt;
            end
        elseif t_clamped <= t_waypoints(1) + dt/2
            % At start of trajectory - use forward difference
            for i = 1:3
                x_future = interp1(t_waypoints, waypoints(:, i+1), t_clamped + dt, params.interpolation_method);
                x_past = x_des(i);
                x_des(i+3) = (x_future - x_past) / dt;
            end
        else
            % Middle of trajectory - use central difference
            for i = 1:3
                x_future = interp1(t_waypoints, waypoints(:, i+1), t_clamped + dt/2, params.interpolation_method);
                x_past = interp1(t_waypoints, waypoints(:, i+1), t_clamped - dt/2, params.interpolation_method);
                x_des(i+3) = (x_future - x_past) / dt;
            end
        end
        
    elseif n_cols == 7 % Both positions and velocities provided - use directly
        for i = 1:6
            x_des(i) = interp1(t_waypoints, waypoints(:, i+1), t_clamped, params.interpolation_method);
        end
    else
        error('trajectory_waypoints must have 4 columns [time,x,y,theta] or 7 columns [time,x,y,theta,dx,dy,dtheta]');
    end
end