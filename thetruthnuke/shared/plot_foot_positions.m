function plot_foot_positions(t, foot_out)
    % Actual foot positions
    Lx = foot_out(:,1);   Ly = foot_out(:,2);
    Rx = foot_out(:,3);   Ry = foot_out(:,4);

    % Actual velocities
    Lx_dot = foot_out(:,5); Ly_dot = foot_out(:,6);
    Rx_dot = foot_out(:,7); Ry_dot = foot_out(:,8);

    % Desired trajectory
    Xd     = foot_out(:,9);
    Yd     = foot_out(:,10);
    Xd_dot = foot_out(:,11);
    Yd_dot = foot_out(:,12);

    figure('Name','Foot Tracking');

    % x position
    subplot(4,1,1); hold on;
    plot(t, Xd,     'k-',  'LineWidth', 2);     % desired
    plot(t, Lx,     'b--', 'LineWidth', 1.2);   % left foot
    plot(t, Rx,     'r--', 'LineWidth', 1.2);   % right foot
    ylabel('X [m]');
    title('Foot X Position vs Time');
    legend('Desired','Left Foot','Right Foot');
    grid on;

    % y position
    subplot(4,1,2); hold on;
    plot(t, Yd,     'k-',  'LineWidth', 2);
    plot(t, Ly,     'b--', 'LineWidth', 1.2);
    plot(t, Ry,     'r--', 'LineWidth', 1.2);
    ylabel('Y [m]');
    title('Foot Y Position vs Time');
    legend('Desired','Left Foot','Right Foot');
    grid on;

    % Xdot
    subplot(4,1,3); hold on;
    plot(t, Xd_dot, 'k-',  'LineWidth', 2);
    plot(t, Lx_dot, 'b--', 'LineWidth', 1.2);
    plot(t, Rx_dot, 'r--', 'LineWidth', 1.2);
    ylabel('dX/dt [m/s]');
    title('Foot X Velocity vs Time');
    legend('Desired','Left Foot','Right Foot');
    grid on;

    % ydot
    subplot(4,1,4); hold on;
    plot(t, Yd_dot, 'k-',  'LineWidth', 2);
    plot(t, Ly_dot, 'b--', 'LineWidth', 1.2);
    plot(t, Ry_dot, 'r--', 'LineWidth', 1.2);
    ylabel('dY/dt [m/s]');
    xlabel('Time [s]');
    title('Foot Y Velocity vs Time');
    legend('Desired','Left Foot','Right Foot');
    grid on;

end