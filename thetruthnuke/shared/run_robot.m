function [t, state_out, cntl_out, foot_out, sim] = run_robot(varargin)
    %Runs the MuJoCo simulation, implementing a Zero-Order Hold (ZOH)
    % for the MPC controller
    global params
    
    % extract params
    p = inputParser;
    addParameter(p, 'simulation_duration', 2, @isnumeric);
    addParameter(p, 'x_0', [], @isnumeric);
    addParameter(p, 'xml_filename', '', @ischar);
    addParameter(p, 'record_video', false, @islogical);
    addParameter(p, 'video_filename', 'mujoco_sim.mp4', @ischar);
    addParameter(p, 'controller_name', @ischar);
    addParameter(p, 'video_fps', 30, @isnumeric);
    parse(p, varargin{:});

    % Python Path Setup
    parentFolder = fileparts(fileparts(mfilename('fullpath')));
    sharedFolder = fullfile(parentFolder, "shared");   % sibling folder named 'shared'
    
    % Add shared folder to Python path if not already present
    if count(py.sys.path, string(sharedFolder)) == 0
        insert(py.sys.path, int32(0), sharedFolder);
    end
    py.importlib.invalidate_caches();
    
    % Import mj_sim from shared
    mj_sim = py.importlib.import_module('mj_sim');
    py.importlib.reload(mj_sim);

    % create sim object
    sim = mj_sim.MujocoSim(p.Results.xml_filename);
    sim.launch_viewer();

    % set inital state
    nx = double(sim.get_nx()); % get # states
    x_0 = p.Results.x_0; % grab IC
    if ~isempty(x_0)
        assert(length(x_0) == nx, 'state0 vector size mismatch. Expected %d elements.', nx);
        sim.set_state_vector(py.numpy.array(x_0.')); % pass state vector to mujoco
    end

    % Simulation Setup
    dt = double(sim.get_timestep());
    simulation_duration = p.Results.simulation_duration;
    num_steps = floor(simulation_duration / dt);
    
    % preallocate outputs
    state_out = zeros(nx, num_steps);
    t = zeros(num_steps, 1);
    cntl_out = zeros(num_steps, 4);
    foot_out = zeros(num_steps, 12);
    
    % start video recording
    if p.Results.record_video
        sim.start_recording(pyargs('filename', p.Results.video_filename, ...
                                   'fps', int32(p.Results.video_fps)));
        frame_interval = 1/p.Results.video_fps;
        frame_time = 0;
    end

    % Zero order hold 
    prev_cntl_t = -params.dt; 
    cntl = zeros(4, 1); % Initialize the control input

    % SIM LOOP
    for i = 1:num_steps
        if ~sim.is_viewer_running()
            disp('Viewer closed early, stopping simulation.');
            break;
        end
        
        x = double(sim.get_state()); % extract state
        state_out(:, i) = x; % store state
        t(i) = double(sim.get_time()); % store time

        % set camera view
        sim.set_camera(pyargs('elevation', -90, ...
                          'distance', 2, ...
                          'lookat', py.list({x(1),x(2)-0.15,x(3)})));

        % Capture frame for video
        if p.Results.record_video && t(i) >= frame_time
            sim.capture_frame(t(i));
            frame_time = frame_time + frame_interval;
        end
        
        % ZOH
        if t(i) >= prev_cntl_t + params.dt - 1e-9 %if time has passed mpc dt
            if nargout(p.Results.controller_name) == 2
                [cntl, foot] = feval(p.Results.controller_name, x, t(i));
            else
                cntl = feval(p.Results.controller_name, x, t(i));
                foot = zeros(1, 12); %handle case without walking
            end
            prev_cntl_t = t(i); %update time of control action
        end

        foot_out(i,:) = foot; %ZOH
        cntl_out(i,:) = cntl'; %otherwise assign the current control to prev control

        %saturate torques and check joint constraints
        [saturated_ctrl, violation] = enforce_constraints(cntl, x, true);

        % if violation
        %     disp('Constraint violation detected, stopping simulation.');
        %     break; % exit loop but keep MATLAB running
        % end

        % Update cntl with saturated value for next iteration and logging
        cntl = saturated_ctrl;
        cntl_out(i,:) = cntl'; % Store the actual applied (saturated) control

        sim.step(saturated_ctrl); %step forward with Mujoco

    end

    if p.Results.record_video
        sim.stop_recording();
    end

    % trim arrays to correct size if ends early
    t = t(1:i);
    state_out = state_out(:, 1:i);
    cntl_out = cntl_out(1:i, :);
    foot_out = foot_out(1:i, :);
    sim.close_viewer();
end