function [cntl, foot_out] = obstacle_course_supervisor(full_state, t)
    global params
    
    % Define phases of the obstacle course
    persistent current_phase
    if isempty(current_phase)
        current_phase = 'running_initial';
    end

    % Define obstacle course parameters (placeholder values)
    params.jump_prep_time = 0.5; % seconds to prepare for a jump
    params.block_x_start = 2.0; % x-position where the block starts
    params.block_height = 0.3; % height of the block
    params.block_x_end = 3.0; % x-position where the block ends
    params.jump_down_prep_time = 0.5; % seconds to prepare for jumping down
    params.gap_x_start = 4.0; % x-position where the gap starts
    params.gap_x_end = 4.5; % x-position where the gap ends
    params.jump_gap_prep_time = 0.5; % seconds to prepare for jumping across gap
    params.finish_line_x = 5.0; % x-position of the finish line

    % Robot's current x position
    robot_x = full_state(1);

    % Initialize control output
    cntl = zeros(4,1);
    foot_out = zeros(1,12);

    switch current_phase
        case 'running_initial'
            if robot_x < 1.8
                [cntl, foot_out] = running_controller(full_state, t);
            else
                disp('Transitioning to preparing_for_jump_up_block');
                current_phase = 'preparing_for_jump_up_block';
                params.t_phase_start = t; % Mark the start time of this new phase
            end
            
        case 'preparing_for_jump_up_block'
            % Stop and prepare for jumping up the block
            if (t - params.t_phase_start) < params.jump_prep_time
                % Call squatting controller or similar preparation here
                cntl = zeros(4,1); % Placeholder: no movement
            else
                disp('Transitioning to jumping_up_block');
                current_phase = 'jumping_up_block';
                params.t_phase_start = t;
            end

        case 'jumping_up_block'
            % Jump up the block
            [cntl, foot_out] = jump_up_block_controller(full_state, t);
            % Transition logic: needs to detect landing on the block
            if full_state(2) > params.block_height - 0.1 && abs(full_state(9)) < 0.1 % Detect landing by y-position and vertical velocity
                disp('Transitioning to walking_on_block');
                current_phase = 'walking_on_block';
                params.t_phase_start = t;
            end

        case 'walking_on_block'
            % Walk on the block until its end
            [cntl, foot_out] = walking_forward_controller(full_state, t);
            if robot_x > params.block_x_end
                disp('Transitioning to preparing_for_jump_down');
                current_phase = 'preparing_for_jump_down';
                params.t_phase_start = t;
            end
            
        case 'preparing_for_jump_down'
            % Stop and prepare for jumping down
            if (t - params.t_phase_start) < params.jump_down_prep_time
                % Call squatting controller or similar preparation here
                cntl = zeros(4,1); % Placeholder: no movement
            else
                disp('Transitioning to jumping_down');
                current_phase = 'jumping_down';
                params.t_phase_start = t;
            end

        case 'jumping_down'
            % Jump down from the block
            [cntl, foot_out] = jump_down_controller(full_state, t);
            % Transition logic: needs to detect landing on the ground after jump down
            if full_state(2) < 0.1 && abs(full_state(9)) < 0.1 && robot_x > params.block_x_end % Detect landing
                disp('Transitioning to running_to_gap');
                current_phase = 'running_to_gap';
                params.t_phase_start = t;
            end

        case 'running_to_gap'
            % Run towards the gap
            [cntl, foot_out] = running_controller(full_state, t);
            if robot_x > params.gap_x_start - 0.5
                disp('Transitioning to preparing_for_jump_across_gap');
                current_phase = 'preparing_for_jump_across_gap';
                params.t_phase_start = t;
            end

        case 'preparing_for_jump_across_gap'
            % Prepare to jump across the gap
            if (t - params.t_phase_start) < params.jump_gap_prep_time
                % Call squatting controller or similar preparation here
                cntl = zeros(4,1); % Placeholder: no movement
            else
                disp('Transitioning to jumping_across_gap');
                current_phase = 'jumping_across_gap';
                params.t_phase_start = t;
            end

        case 'jumping_across_gap'
            % Jump across the gap
            [cntl, foot_out] = jump_across_gap_controller(full_state, t);
            % Transition logic: needs to detect landing after jumping the gap
            if full_state(2) < 0.1 && abs(full_state(9)) < 0.1 && robot_x > params.gap_x_end + 0.5 % Simplified landing detection
                disp('Transitioning to running_to_finish');
                current_phase = 'running_to_finish';
                params.t_phase_start = t;
            end

        case 'running_to_finish'
            % Run to the finish line
            [cntl, foot_out] = running_controller(full_state, t);
            if robot_x > params.finish_line_x
                disp('Obstacle course completed!');
                current_phase = 'finished';
                params.t_phase_start = t;
            end
            
        case 'finished'
            cntl = zeros(4,1); % Stop the robot
            foot_out = zeros(1,12);
            
        otherwise
            cntl = zeros(4,1);
            foot_out = zeros(1,12);
    end

    % You'll need to pass the current_phase to the animate_robot function
    % to visualize the phase transitions.
end
