clear, clc, close all
s = settings;
s.matlab.appearance.figure.GraphicsTheme.TemporaryValue = 'light';
addpath(fullfile(fileparts(pwd), "shared"));

%% Initial Conditions and Parameters
xml_file = 'Robot_Basic_Task_2.xml';
video_filename = 'walking_forward.mp4';
controller_name = 'walking_forward_controller';

% global params
global params
params.l = 0.22;
params.a = 0.25;
params.b = 0.15;
params.m = 8;
params.I = (1/12) * (params.a^2 + params.b^2) * params.m;
params.mu = 0.7;
params.g = 9.81;

params.dx_des = 0.5; %[m/s]
params.ypos_des = 0.5; %[m]

% Build IC state
q_pos0 = [0; 0.52; 0; pi/8; pi/3; -pi/4; pi/3];
q_vel0 = zeros(7,1);
x_0 = [q_pos0; q_vel0];

%% controller design
% constraints
params.mu = 0.5;
params.Fy_constraint = [-10 250];

% tuning
params.N = 20;
params.dt = 0.02;

            %    [x, y, theta, dx, dy, dtheta, g]
params.Q = diag([200 200 800 30 10 30 0]);
params.R = 1e-4 * diag([1, 1, 1, 1]);

% swing control
params.Kv = 0.3;
params.swing_height = 0.03;
params.Kp = diag([100, 200]);
params.Kd = eye(2) * 5;

%% run
[t, state_out, cntl_out, foot_out, sim] = run_robot('simulation_duration', 6, ...
    'x_0', x_0, ...
    'xml_filename', xml_file,...
    'video_filename', video_filename,...
    'record_video', true,...
    'controller_name', controller_name);

%% plot
plot_outputs(t, state_out, cntl_out)
plot_foot_positions(t, foot_out)
