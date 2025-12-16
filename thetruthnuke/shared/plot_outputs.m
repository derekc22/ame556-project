function plot_outputs(t, state_out, cntl_out)
    set(groot, 'defaultTextInterpreter', 'latex');
    set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
    set(groot, 'defaultLegendInterpreter', 'latex');
    
    plot_state_pos(t, state_out); %plot state positions
    plot_joint_velocities(t, state_out) % plot state velocities
    plot_cntl(t, cntl_out) % plot cntl

end

%% helpers

function plot_state_pos(t, state_out)
    global limits

    % qpos = [x; y; theta; q1; q2; q3; q4]
    x = state_out(1, :);
    y = state_out(2, :);
    theta = state_out(3, :);
    q = state_out(4:7, :);  % joint angles q1-q4
    
    % Define consistent colors
    colors = get_joint_colors();
    body_colors = get_body_colors();
    
    % Positions subplot
    figure('Name','Positions');
    subplot(2,1,1); hold on;
    plot(t, x, 'LineWidth', 1.5, 'Color', body_colors.x);
    plot(t, y, 'LineWidth', 1.5, 'Color', body_colors.y);
    ylabel('Position [m]');
    xlabel('Time [s]');
    legend('x', 'y');
    grid on; box off;
    
    % Angles subplot with limits
    subplot(2,1,2); hold on;
    plot(t, theta, 'LineWidth', 1.5, 'Color', body_colors.theta);
    
    % Plot joints and limits
    for j = 1:4
        plot(t, q(j,:), 'LineWidth', 1.5, 'Color', colors(j,:));
    end
    
    % Add limit lines (only show in legend once per pair)
    yline(limits.angle_max(1), '--', 'Color', colors(1,:), ...
          'LineWidth', 1.2, 'Alpha', 0.5, 'DisplayName', 'q_{1,3} lim');
    yline(limits.angle_min(1), '--', 'Color', colors(1,:), ...
          'LineWidth', 1.2, 'Alpha', 0.5, 'HandleVisibility','off');
    
    yline(limits.angle_max(2), '-.', 'Color', colors(2,:), ...
          'LineWidth', 1.2, 'Alpha', 0.5, 'DisplayName', 'q_{2,4} lim');
    yline(limits.angle_min(2), '-.', 'Color', colors(2,:), ...
          'LineWidth', 1.2, 'Alpha', 0.5, 'HandleVisibility','off');
    
    ylabel('Angles [rad]');
    xlabel('Time [s]');
    legend('\theta','q_1','q_2','q_3','q_4','q_{1,3} lim','q_{2,4} lim', 'Location','best');
    grid on; box off;
end

function plot_cntl(t, cntl_out)
    global limits
    tau = cntl_out; % Nx4
    
    % Use consistent colors
    colors = get_joint_colors();
    
    figure('Name','Torques'); hold on;
    
    % Plot torques
    for j = 1:4
        plot(t, tau(:,j), 'LineWidth', 1.5, 'Color', colors(j,:));
    end
    
    % Add limit lines (only show in legend once per pair)
    yline(limits.cntl_lim(1), '--', 'Color', colors(1,:), ...
          'LineWidth', 1.2, 'Alpha', 0.5, 'DisplayName', '\tau_{1,3} lim');
    yline(-limits.cntl_lim(1), '--', 'Color', colors(1,:), ...
          'LineWidth', 1.2, 'Alpha', 0.5, 'HandleVisibility','off');
  
    yline(limits.cntl_lim(2), '-.', 'Color', colors(2,:), ...
          'LineWidth', 1.2, 'Alpha', 0.5, 'DisplayName', '\tau_{2,4} lim');
    yline(-limits.cntl_lim(2), '-.', 'Color', colors(2,:), ...
          'LineWidth', 1.2, 'Alpha', 0.5, 'HandleVisibility','off');
 
    ylabel('Torques [Nm]');
    xlabel('Time [s]');
    legend('\tau_1','\tau_2','\tau_3','\tau_4','\tau_{1,3} lim','\tau_{2,4} lim', 'Location','best');
    grid on; box off;
end

function plot_joint_velocities(t, state_out)
    global limits
    dq = state_out(11:14, :);    % joint velocities q1–q4
    body_vel = state_out(8:10,:); % [dx; dy; dtheta]
    
    colors = get_joint_colors();
    body_colors = get_body_colors();
    
    figure('Name','Joint Vel');
    tiledlayout(2,1);
    nexttile; hold on;
    
    plot(t, body_vel(3,:), 'LineWidth', 1.5, 'Color', body_colors.theta, ...
         'DisplayName', 'd\theta');
    
    % Plot joint velocities
    for j = 1:4
        plot(t, dq(j,:), 'LineWidth', 1.5, 'Color', colors(j,:), ...
             'DisplayName', sprintf('dq_%d', j));
    end
    
    % q1/q3 limits
    yline(limits.vel_lim(1), '--', 'Color', colors(1,:), ...
          'LineWidth', 1.2, 'Alpha', 0.6, 'DisplayName', 'dq_{1,3} lim');
    yline(-limits.vel_lim(1), '--', 'Color', colors(1,:), ...
          'LineWidth', 1.2, 'Alpha', 0.6, 'HandleVisibility', 'off');

    % q2/q4 limits
    yline(limits.vel_lim(2), '-.', 'Color', colors(2,:), ...
          'LineWidth', 1.2, 'Alpha', 0.6, 'DisplayName', 'dq_{2,4} lim');
    yline(-limits.vel_lim(2), '-.', 'Color', colors(2,:), ...
          'LineWidth', 1.2, 'Alpha', 0.6, 'HandleVisibility', 'off');

    xlabel('Time [s]');
    ylabel('Angular Velocity [rad/s]');
    legend('Location','best');
    grid on; box off;
    
    % Body translational velocities
    nexttile; hold on;
    plot(t, body_vel(1,:), 'LineWidth', 1.5, 'Color', body_colors.x);  % dx
    plot(t, body_vel(2,:), 'LineWidth', 1.5, 'Color', body_colors.y);  % dy
    xlabel('Time [s]');
    ylabel('Body Velocity [m/s]');
    legend('dx','dy','Location','best');
    grid on; box off;
end

function colors = get_joint_colors()
    colors = [
        0.0000, 0.4470, 0.7410;  % Blue (q1)
        0.8500, 0.3250, 0.0980;  % Orange (q2)
        0.9290, 0.6940, 0.1250;  % Yellow (q3)
        0.4940, 0.1840, 0.5560;  % Purple (q4)
    ];
end

function body_colors = get_body_colors()
    body_colors.x = [0.8500, 0.3250, 0.0980];     % Orange for x/dx
    body_colors.y = [0.4660, 0.6740, 0.1880];     % Green for y/dy
    body_colors.theta = [0, 0, 0]; % Pblack theta/dtheta
end