function animate_robot(t, state_out, Fs)
    % t         : time vector
    % state_out : [x; y; theta; q1; q2; q3; q4] x N
    % Fs        : animation frame rate
    global params

    % Extract states
    x = state_out(1, :);
    y = state_out(2, :);
    theta = state_out(3, :);
    q = state_out(4:7, :);

    % Resample for smooth animation
    [t_e, x_e] = even_sample(t, [x; y; theta; q], Fs);
    
    fig_anim = figure('Color','w');
    ax_anim = gca;
    axis equal;
    grid on;
    xlabel('X [m]'); ylabel('Y [m]');
    xlim([min(x)-0.3, max(x)+0.3]);
    ylim([min(y)-0.5, max(y)+0.3]);

    % Setup video writer
    spwriter = VideoWriter('robot_animation.avi','Motion JPEG AVI');
    spwriter.FrameRate = Fs;
    open(spwriter);

    for k = 1:length(t_e)
        cla(ax_anim); hold(ax_anim, 'on');
        
        body_coords = x_e(1:3,k);   % [x; y; theta]
        qk = x_e(4:7,k);            % [q1;q2;q3;q4]

        FK = robot_FK(body_coords, qk, params.l, params.a);

        % Draw trunk
        draw_box(body_coords(1:2), params.b, params.a, body_coords(3), 'k');

        % Left leg
        plot([FK.hinge(1), FK.p1(1)], [FK.hinge(2), FK.p1(2)], 'b', 'LineWidth',3);
        plot([FK.p1(1), FK.p2(1)], [FK.p1(2), FK.p2(2)], 'r', 'LineWidth',3);

        % Right leg
        plot([FK.hinge(1), FK.p3(1)], [FK.hinge(2), FK.p3(2)], 'g', 'LineWidth',3);
        plot([FK.p3(1), FK.p4(1)], [FK.p3(2), FK.p4(2)], 'y', 'LineWidth',3);

        %ground plane
        plot(ax_anim, [-1, 1]*10, [0 0], 'k', 'LineWidth', 1);

        title(sprintf('Time: %.2f s', t_e(k)));
        drawnow;

        % Write frame
        frame = getframe(fig_anim);
        writeVideo(spwriter, frame);
    end

    close(spwriter);
end

%% Helper functions
function draw_box(center, width, height, theta, color)
    % Draws a rectangle (trunk)
    corners = [ width/2  width/2 -width/2 -width/2;
                height/2 -height/2 -height/2  height/2;
                1        1         1         1];
    T = [cos(theta) -sin(theta) center(1);
         sin(theta)  cos(theta) center(2);
         0           0          1];
    rotated = T * corners;
    fill(rotated(1,:), rotated(2,:), color, 'FaceAlpha',0.5,'EdgeColor','k','LineWidth',1.2);
end

function [Et, Ex] = even_sample(t, x, Fs)
    % Interpolate to even time steps
    Et = (t(1):1/Fs:t(end))';
    Ex = interp1(t, x', Et)';
end
