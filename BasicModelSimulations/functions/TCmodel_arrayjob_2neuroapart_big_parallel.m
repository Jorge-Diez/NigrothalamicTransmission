
% simulation done supposing that there is only 1 core performing
% each experiment, done for the iceberg HPC
function [dir_name] = TCmodel_arrayjob_2neuroapart_big_parallel(job_id, num_jobs)

% in this experiment deletion method is used but
% 2 neurons are always "apart" using a different motherspike
% difference is small, but prefer to have different files to avoid
% confusion




%% Parameter creation and sim mode
sim_mode = 3;


F_SNr = 51:1:90; %firing rate increase
nr_neurons = 1:1:28; %nr of neurons with firing rate increase
FREQ_NR_COMB = combvec(nr_neurons, F_SNr);




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

corr_CX = 0;  %correlation of exc inputs
corr_SNr = 0; % correlation of nigral inputs

G_SNr_all = 0.70;  %GS (conductances) %CHANGED
G_SNr = G_SNr_all;
num_trials = 10;   %NT  THIS IS CHANGED FROM 100 TO 10, SHOULD CHANGE TO 100 FOR GOOD RESULTS (LONG!)


%% Loading data

corr_vals = 0.3:0.1:1;   %values of correlation among inhibitory inputs

NT_GS_JV_TF_all = combvec(G_SNr,corr_vals); %matrix of all possible combinations of conductances and corr values
ALL_EXPERIMENTS = combvec(NT_GS_JV_TF_all, FREQ_NR_COMB);
%use number of threads available as MAX

NT_GS_JV_TF = ALL_EXPERIMENTS(:,job_id:num_jobs:end); %parameter space

% comb_trial_num = NT_GS_JV_TF(1,:);
G_SNr = NT_GS_JV_TF(1,:); %extract conductance value
jit_val = NT_GS_JV_TF(2,:); %extract correlation value
nr_neurons_increase = NT_GS_JV_TF(3,:); %extract number of neurons with freq increase
freq_increase = NT_GS_JV_TF(4,:); % extract the frequency increase value

rebound_spk = zeros(8960/num_jobs,num_trials); %Result vector, rebound spikes after decrease
all_reb_spk = zeros(8960/num_jobs,num_trials); %Result vector, (check how its done may be the whole simuation)


for exp = 1:size(NT_GS_JV_TF,2)
    R2 = zeros(size(NT_GS_JV_TF,2),num_trials); %used to store the correlation values
    
    
    [F_SNr, FG_SNR] = par_obtain(nr_neurons_increase(exp), freq_increase(exp), 28);
    
    SPK = [];
    
    exppath = ['ICEBERG_DELET_2NEURONS/FREQINCTO_',num2str(freq_increase(exp))];
    freq_experiment = fullfile(pwd, exppath);
    if exist(freq_experiment,'dir') ~= 7
        mkdir(freq_experiment)
    end
    
    specific_folder = ['nr_neurons_',num2str(nr_neurons_increase(exp))];
    dir_name = fullfile(freq_experiment,specific_folder );
    
    dir_name_trace = [dir_name,'/voltage-traces-and-inputs/'];
    if exist(dir_name_trace,'dir') ~= 7
        mkdir(dir_name_trace)
    end
    
    
    
    %perform experiment
    
    
    switch sim_mode %for now we only have sim mode 1 and 2
        
        case 1
            
            [rebound_spk(exp,:),all_reb_spk(exp,:)] = ...
                TC_model_CX_SNr_cond_changed_parfor_opt_diff_mothspikes(...
                F_SNr,0,G_SNr(exp),...
                T,mov_onset,jit_val(exp),...
                num_trials,dir_name_trace,exp,FG_SNR);
        case 2
            
            [rebound_spk(exp,:),all_reb_spk(exp,:)] = ...
                TC_model_CX_SNr_cond_changed_parfor_opt(...
                F_SNr,0,G_SNr(exp),...
                T,mov_onset,jit_val(exp),...
                num_trials,dir_name_trace,exp,FG_SNR);
            
        case 3
            
            [rebound_spk(exp,:),all_reb_spk(exp,:)] = ...
                TC_model_CX_SNr_cond_changed_parfor_opt_2neurons(...
                F_SNr,0,G_SNr(exp),...
                T,mov_onset,jit_val(exp),...
                num_trials,dir_name_trace,exp,FG_SNR);
            
            
            
    end
    
    disp(['subexperiment ', num2str(exp), ' has finished'])
    
    
    
    
    
    
end

% save results
for i = 1:length(freq_increase)
    
    
    exppath = ['ICEBERG_DELET_2NEURONS/FREQINCTO_',num2str(freq_increase(i))];
    freq_experiment = fullfile(pwd, exppath);

    
    specific_folder = ['nr_neurons_',num2str(nr_neurons_increase(i))];
    dir_name = fullfile(freq_experiment,specific_folder );
    


    dir_name_cp = [dir_name,'/res-for-colorplot/'];
    
    if exist(dir_name_cp,'dir') ~= 7
        mkdir(dir_name_cp)
    end
    
    
    [F_SNr, FG_SNR] = par_obtain(nr_neurons_increase(i), freq_increase(i), 28);
    
    temp_rebound_spk = rebound_spk(i,:);
    temp_all_reb_spk = all_reb_spk(i,:);
    temp_G_SNR = G_SNr(i);
    temp_NT_GS_JV_TF = NT_GS_JV_TF(:,i);
    temp_F_SNr = F_SNr;
    temp_FG_SNR = FG_SNR;
    
    
    save([dir_name_cp 'corr-' num2str(jit_val(i)*100) '-freq_inc-' num2str(freq_increase(i)) '-nSNr_inc-' num2str(nr_neurons_increase(i)) '-job-' num2str(job_id)],...
        'temp_rebound_spk','temp_all_reb_spk','temp_G_SNR',...
        'num_trials','temp_NT_GS_JV_TF','temp_F_SNr','temp_FG_SNR')
    
    
    
end





end
% exit
