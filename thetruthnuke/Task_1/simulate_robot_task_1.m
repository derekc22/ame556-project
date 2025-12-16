clear, clc, close all
s = settings;
s.matlab.appearance.figure.GraphicsTheme.TemporaryValue = 'light';
addpath(fullfile(fileparts(pwd), "shared"));

%% Initial Conditions and Parameters
xml_file = 'Robot_Basic_Task_1.xml';
video_filename = 'task_1.mp4';
controller_name = 'mpc_controller_task_1';

% global params
global params
params.l = 0.22;
params.a = 0.25;
params.b = 0.15;
params.m = 8;
params.I = (1/12) * (params.a^2 + params.b^2) * params.m;
params.mu = 0.7;
params.g = 9.81;

%% Define Trajectory Waypoints
% 1) Position only: [time, x, y, theta] - velocities auto-derived
% 2) Full state: [time, x, y, theta, dx, dy, dtheta] - explicit control

params.trajectory_waypoints = [
    % squatting
    0.0,  0.0,  0.45,  0;
    1.0,  0.0,  0.45,  0; % settle time 
    1.5,  0.0,  0.55,  0; %stand
    2,  0.0,  0.55,  0;
    3,  0,  0.4,  0; %squat
    4,  0,  0.4,  0; 
];

% Set interpolation method: 'linear', 'pchip', 'spline', or 'makima'
params.interpolation_method = 'pchip';

% Build IC state
q_pos0 = [0; 0.6; 0; -4*pi/12; pi/2; -pi/7; pi/2];
q_vel0 = zeros(7,1);
x_0 = [q_pos0; q_vel0];

%% MPC controller design
% constraints
params.mu = 0.5;

% tuning
params.N = 30;
params.dt = 0.02;

            %    x, y, theta, dx, dy, dtheta, g
params.Q = diag([400 200 1000 10 10 10 0]);
params.R = 1e-4 * diag([1, 1, 1, 1]);

%% run
[t, state_out, cntl_out, sim] = run_robot('simulation_duration', 4, ...
    'x_0', x_0, ...
    'xml_filename', xml_file,...
    'video_filename', video_filename,...
    'record_video', true,...
    'controller_name', controller_name);

%% plot
plot_outputs(t, state_out, cntl_out)
plot_trajectory_comparison(t, state_out)