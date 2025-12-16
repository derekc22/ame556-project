function cntl = jump_across_gap_controller(full_state, t)
    global params

    % This controller will be responsible for making the robot jump across a gap.
    % Similar to the jump_up_block controller, it will involve:
    % 1. Defining a desired trajectory for the CoM and foot positions suitable for traversing a gap.
    % 2. Implementing a control strategy to follow this trajectory.

    % Placeholder for now: return zero control
    cntl = zeros(4,1);
end
