% simulation done supposing that there is only 1 core performing
% each experiment, done for the iceberg HPC
function [dir_name] = TCmodel_threegroup






%% Parameter creation and sim mode
sim_mode = 2;


F_SNr = [50 20 82];
FG_SNR = [3 9 18]; %nr of neurons with firing rate increase



%% Ranges of variations for both CX and SNr
dt = 0.01;
simtime = 1500; %(ms)
T = 0:dt:simtime;

mov_onset = 1000; %decrease in firing rate at t = this

%N_CX = 200; %number of exc inputs
N_SNr = 30; %number of nigral inputs

%F_CX = 1:0.5:10; %firing rate of exc inputs
%F_SNr = 50; %firing rate of nigral inputs

%n_trials_var = length(N_CX);

% F_CX = 10*ones(size(T));
% F_SNr = 80*sigmf(T,[-0.1 mov_onset]);

G_SNr_all = 0.70;  %GS (conductances) %CHANGED
G_SNr = G_SNr_all;
num_trials = 10;   %NT  THIS IS CHANGED FROM 100 TO 10, SHOULD CHANGE TO 100 FOR GOOD RESULTS (LONG!)


%% Loading data

corr_vals = 0.3:0.1:1;   %values of correlation among inhibitory inputs

NT_GS_JV_TF_all = combvec(G_SNr,corr_vals); %matrix of all possible combinations of conductances and corr values
%use number of threads available as MAX



rebound_spk = zeros(1,num_trials); %Result vector, rebound spikes after decrease
all_reb_spk = zeros(1,num_trials); %Result vector, (check how its done may be the whole simuation)





exppath = ['DELETMETHOD_THREEGROUP_C1'];
dir_name = fullfile(pwd, exppath);
if exist(dir_name,'dir') ~= 7
    mkdir(dir_name)
end


dir_name_trace = [dir_name,'/voltage-traces-and-inputs/'];
if exist(dir_name_trace,'dir') ~= 7
    mkdir(dir_name_trace)
end



%perform experiment

parfor S = 1:size(NT_GS_JV_TF_all,2)
    switch sim_mode %for now we only have sim mode 1 and 2
        
        case 1
            
            [rebound_spk(S,:),all_reb_spk(S,:)] = ...
                TC_model_CX_SNr_cond_changed_parfor_opt_diff_mothspikes(...
                F_SNr,0,G_SNr,...
                T,mov_onset,corr_vals(S),...
                num_trials,dir_name_trace,S,FG_SNR);
        case 2
            
            [rebound_spk(S,:),all_reb_spk(S,:)] = ...
                TC_model_CX_SNr_cond_changed_parfor_opt(...
                F_SNr,0,G_SNr,...
                T,mov_onset,corr_vals(S),...
                num_trials,dir_name_trace,S,FG_SNR);
            
            
    end
    
    disp(['subexperiment ', num2str(S), ' has finished'])
    
end



dir_name_cp = [dir_name,'/res-for-colorplot/'];

if exist(dir_name_cp,'dir') ~= 7
    mkdir(dir_name_cp)
end


save([dir_name_cp '-nSNr-' num2str(sum(FG_SNR))],...
    'rebound_spk','all_reb_spk','G_SNr',...
    'num_trials','NT_GS_JV_TF_all','F_SNr','FG_SNR', 'corr_vals')







end
% exit
