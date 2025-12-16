function plot_trajectory_comparison(t, state_out)
    global params
    
    % Ensure correct orientation (rows = time points, cols = states)
    if size(state_out, 1) < size(state_out, 2)
        state_out = state_out';
    end
    
    % Ensure t is a column vector
    if size(t, 2) > size(t, 1)
        t = t';
    end
    
    n_points = length(t);
    
    % Preallocate arrays for desired states
    x_des = zeros(n_points, 1);
    y_des = zeros(n_points, 1);
    theta_des = zeros(n_points, 1);
    dx_des = zeros(n_points, 1);
    dy_des = zeros(n_points, 1);
    dtheta_des = zeros(n_points, 1);
    
    % Get desired state at each time point
    for i = 1:n_points
        des_state = get_desired_state(t(i));
        x_des(i) = des_state(1);
        y_des(i) = des_state(2);
        theta_des(i) = des_state(3);
        dx_des(i) = des_state(4);
        dy_des(i) = des_state(5);
        dtheta_des(i) = des_state(6);
    end
    
    % Extract actual states
    x_act = state_out(:, 1);
    y_act = state_out(:, 2);
    theta_act = state_out(:, 3);
    dx_act = state_out(:, 8);
    dy_act = state_out(:, 9);
    dtheta_act = state_out(:, 10);
    
    % Get waypoints
    waypoints = params.trajectory_waypoints;
    
    % Get body colors
    body_colors = get_body_colors();
    
    figure('Name', 'Position Tracking');
    
    % X Position
    subplot(3, 1, 1); hold on;
    plot(t, x_act, 'LineWidth', 1.5, 'Color', body_colors.x);
    plot(t, x_des, 'r--', 'LineWidth', 1.5);
    if size(waypoints, 2) >= 4
        plot(waypoints(:, 1), waypoints(:, 2), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
    end
    grid on; box off;
    xlabel('Time [s]');
    ylabel('x [m]');
    legend('Actual', 'Desired', 'Waypoints', 'Location', 'best');
    
    % Y Position
    subplot(3, 1, 2); hold on;
    plot(t, y_act, 'LineWidth', 1.5, 'Color', body_colors.y);
    plot(t, y_des, 'r--', 'LineWidth', 1.5);
    if size(waypoints, 2) >= 4
        plot(waypoints(:, 1), waypoints(:, 3), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
    end
    grid on; box off;
    xlabel('Time [s]');
    ylabel('y [m]');
    legend('Actual', 'Desired', 'Waypoints', 'Location', 'best');
    
    % Theta
    subplot(3, 1, 3); hold on;
    plot(t, theta_act*180/pi, 'LineWidth', 1.5, 'Color', body_colors.theta);
    plot(t, theta_des*180/pi, 'r--', 'LineWidth', 1.5);
    if size(waypoints, 2) >= 4
        plot(waypoints(:, 1), waypoints(:, 4)*180/pi, 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
    end
    grid on; box off;
    xlabel('Time [s]');
    ylabel('$\theta$ [deg]');
    legend('Actual', 'Desired', 'Waypoints', 'Location', 'best');
    
    figure('Name', 'Velocity Tracking');
    
    % X Velocity
    subplot(3, 1, 1); hold on;
    plot(t, dx_act, 'LineWidth', 1.5, 'Color', body_colors.x);
    plot(t, dx_des, 'r--', 'LineWidth', 1.5);
    if size(waypoints, 2) == 7
        plot(waypoints(:, 1), waypoints(:, 5), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
    end
    grid on; box off;
    xlabel('Time [s]');
    ylabel('dx [m/s]');
    if size(waypoints, 2) == 7
        legend('Actual', 'Desired', 'Waypoints', 'Location', 'best');
    else
        legend('Actual', 'Desired', 'Location', 'best');
    end
    
    % Y Velocity
    subplot(3, 1, 2); hold on;
    plot(t, dy_act, 'LineWidth', 1.5, 'Color', body_colors.y);
    plot(t, dy_des, 'r--', 'LineWidth', 1.5);
    if size(waypoints, 2) == 7
        plot(waypoints(:, 1), waypoints(:, 6), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
    end
    grid on; box off;
    xlabel('Time [s]');
    ylabel('dy [m/s]');
    if size(waypoints, 2) == 7
        legend('Actual', 'Desired', 'Waypoints', 'Location', 'best');
    else
        legend('Actual', 'Desired', 'Location', 'best');
    end
    
    % Angular Velocity
    subplot(3, 1, 3); hold on;
    plot(t, dtheta_act*180/pi, 'LineWidth', 1.5, 'Color', body_colors.theta);
    plot(t, dtheta_des*180/pi, 'r--', 'LineWidth', 1.5);
    if size(waypoints, 2) == 7
        plot(waypoints(:, 1), waypoints(:, 7)*180/pi, 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
    end
    grid on; box off;
    xlabel('Time [s]');
    ylabel('$\dot{\theta}$ [deg/s]');
    if size(waypoints, 2) == 7
        legend('Actual', 'Desired', 'Waypoints', 'Location', 'best');
    else
        legend('Actual', 'Desired', 'Location', 'best');
    end
end

function body_colors = get_body_colors()
    body_colors.x = [0.8500, 0.3250, 0.0980];     % Orange for x/dx
    body_colors.y = [0.4660, 0.6740, 0.1880];     % Green for y/dy
    body_colors.theta = [0, 0, 0]; % Pblack theta/dtheta
end