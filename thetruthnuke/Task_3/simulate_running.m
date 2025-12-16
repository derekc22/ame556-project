clear, clc, close all
s = settings;
s.matlab.appearance.figure.GraphicsTheme.TemporaryValue = 'light';
addpath(fullfile(fileparts(pwd), "shared"));

%% Initial Conditions and Parameters
xml_file = 'Robot_Basic_Task_3.xml';
video_filename = 'running.mp4';
controller_name = 'running_controller';

% global params
global params
params.l = 0.22;
params.a = 0.25;
params.b = 0.15;
params.m = 8;
params.I = (1/12) * (params.a^2 + params.b^2) * params.m;
params.mu = 0.7;
params.g = 9.81;

% Build IC state
q_pos0 = [0; 0.45; 0; -pi/3; 1.38; -pi/6; pi/2];
q_vel0 = [0.0; 0; 0; 0; 0; 0; 0];
x_0 = [q_pos0; q_vel0];

%% controller design
% constraints
params.mu = 0.5;
params.Fy_constraint = [-10 350];

% tuning
params.N = 30;
params.dt = 0.01;

            %    [x, y, theta, dx, dy, dtheta, g]
params.Q = diag([200 500 900 100 10 30 0]);
params.R = 1e-4 * diag([1, 1, 1.2, 1.2]);

% swing control
params.ypos_des = 0.5; %[m]
params.stance_time = 0.16;
params.swing_height = 0.07;
params.Kp = 0.8*diag([200, 150]);
params.Kd = diag([1,1]) * 6;

%% gain scheduling
params.schedule.dx_des.time =  [0,  1,   2,    3,    5];
params.schedule.dx_des.value = [0,  0.8, 1.55,  1.75,  1.7 ];

params.schedule.theta_des.time = [0, 3, 5];
params.schedule.theta_des.value = deg2rad([0, -5, 0]);

params.schedule.Kv.time = [0, 3, 6];
params.schedule.Kv.value = [0.1, 0.13, 0.12];

%change flight time for first second
params.schedule.flight_time.time = [0, 2];
params.schedule.flight_time.value = [0, 0.03];

params.schedule.swing_height.time = [0, 0.5, 2, 4, 5.5];
params.schedule.swing_height.value = [0.01, 0.04, 0.06, 0.06, 0.04];

params.schedule.stance_time.time = [0, 1];
params.schedule.stance_time.value = [0.05, 0.16];
%% run
[t, state_out, cntl_out, foot_out, sim] = run_robot('simulation_duration', 6.25, ...
    'x_0', x_0, ...
    'xml_filename', xml_file,...
    'video_filename', video_filename,...
    'record_video', true,...
    'controller_name', controller_name,...
    'video_fps', 120);

%% plot
plot_outputs(t, state_out, cntl_out)
plot_foot_positions(t, foot_out)