%% Date created 15.08.18 by M. Mohagheghi

% This script generates the results for the transmission quality for
% exponential and binomial amplitude distributions

addpath('functions')

% Exponential

% Running the simulation; 1st input arg. specifies which chunk in
% parameter space should be run and 2nd arg. specifies how many chunks
% should the parameter space be devided into


mov_onset = 1000; % in ms
N_CX = 200; %Do not contemplate for now

F_CX = 1:0.5:10; %Do not contemplate for now



F_SNr = [71]; %Hz
F_Group_neurons = [30]; %groups of neurons with specific firing rates
N_SNr = sum(F_Group_neurons);
G_SNr_all = 0.70; %nigral conductance
corr_vals = 0:0.05:1; %values of correlation among inhibitory inputs
num_trials = 100;

disp(['Running simulations to compute transmission quality', ...
      ' for binomial amplitude distribution ...'])

disp( datestr(now, 'dd/mm/yy-HH:MM'))
res_dir_mip = TCmodel_func_bwfor(1, 1 , mov_onset, N_CX, N_SNr, F_CX, F_SNr, G_SNr_all, num_trials, corr_vals,F_Group_neurons);  %binomial
disp( datestr(now, 'dd/mm/yy-HH:MM'))
vis_res_lumped_mats(res_dir_mip, 'MIP')




%disp(['Showing the summary results ...'])
%res_summary(res_dir_exp, res_dir_mip)