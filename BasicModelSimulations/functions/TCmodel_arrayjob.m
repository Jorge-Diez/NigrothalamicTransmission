%% Modification date 07.03.16

% Here I modified the script so that it can compute the Pinsky-Rinzel 
% measure of synchrony from the spike trains generated using MIP


%% Date 03.12.15

%% TCmodel_clus_func is function which thoughout the comparison-of-all-3 dir
%% Prepares the intialization like loading experimental data (if necessary)
%% , specifying the parameters required for being sweeped or scanned, doing
%% some analysis on one the experimental data e.g. to extract the trails having
%% almost same rate and same distribution. It also initialize important para-
%% meters for running on clusters.

% The contents of the function mentioned here is to use spike trains
% based on the MIP mode of spike trains. . To have a
% better results and enough inputs the total simulation time is 1500 ms and
% the movement event occurs at 1000 ms.

% The inputs to this function is job_id and num_jobs which split the
% G_SNr parameter to num_jobs smaller one, so it can be distributed along
% clusters. 'test_bashfile.m' and 'bashfile_gen_bwfor.m' are the scripts be
% relevant for this purpose.
%% 08.07.2015 (For Loop optimization)
% job_id = 1;
% num_jobs = 1;

function [dir_name] = TCmodel_arrayjob(job_id, num_jobs)

% The goal of this m-file is to plot the results as Robert wants for his
% reports. In these simulations, there is no cortical inputs and the goal
% is to understand how correlation and conductance of SNr inputs affect
% rebound activity in a thalamocortical cell. Each simulation will be
% repeated 100 times and then as a result, the percentage of reobund
% activity is determined

% sim_mode determines how the different spike times are generated
% 1 - for each group a different mother spike train is generated,
% correlation is not taken into account between different spike trains
% 2 - a unique mother spike train is generated for highest frequency, and
% the group with lower frequency deletes spikes randomly (uniform)





%% Parameter creation and sim mode
    sim_mode = 1;


    F_SNr = 51:1:90; %firing rate increase 
    nr_neurons = 1:1:30; %nr of neurons with firing rate increase
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
    
    rebound_spk = zeros(size(NT_GS_JV_TF,2),num_trials); %Result vector, rebound spikes after decrease
    all_reb_spk = zeros(size(NT_GS_JV_TF,2),num_trials); %Result vector, (check how its done may be the whole simuation)
    
    R2 = zeros(size(NT_GS_JV_TF,2),num_trials); %used to store the correlation values

    % comb_trial_num = NT_GS_JV_TF(1,:);
    G_SNr = NT_GS_JV_TF(1,:); %extract conductance value
    jit_val = NT_GS_JV_TF(2,:); %extract correlation value
    nr_neurons_increase = NT_GS_JV_TF(3,:); %extract number of neurons with freq increase
    freq_increase = NT_GS_JV_TF(4,:); % extract the frequency increase value
    
    
    if (nr_neurons_increase == N_SNr)
        F_SNr = freq_increase; %Hz
        
        FG_SNR = nr_neurons_increase; %groups of neurons with specific firing rates
        
    else
        
        %obtain parameters
        F_SNr = [50 freq_increase]; %Hz
        
        FG_SNR = [N_SNr - nr_neurons_increase nr_neurons_increase]; %groups of neurons with specific firing rates
    end
    
    
    

    SPK = [];
    
    exppath = ['CLUSTERTRY/FREQINCTO_',num2str(freq_increase)];
    freq_experiment = fullfile(pwd, exppath);
        if exist(freq_experiment,'dir') ~= 7
            mkdir(freq_experiment)
        end
    
    specific_folder = ['nr_neurons_',num2str(nr_neurons_increase)];
    dir_name = fullfile(freq_experiment,specific_folder );
    

    dir_name_trace = [dir_name,'/voltage-traces-and-inputs/'];
    if exist(dir_name_trace,'dir') ~= 7
        mkdir(dir_name_trace)
    end

  
    
    for S = 1:size(NT_GS_JV_TF,2)   % Loop over experimental trials
        %disp(['jobnum = ',num2str(job_id), ', S = ',num2str(S)])
        switch sim_mode %for now we only have sim mode 1 and 2
            
            case 1
            
                [rebound_spk(S,:),all_reb_spk(S,:)] = ...
                    TC_model_CX_SNr_cond_changed_parfor_opt_diff_mothspikes(...
                                    F_SNr,0,G_SNr(S),...
                                    T,mov_onset,jit_val(S),...
                                    num_trials,dir_name_trace,S,FG_SNR);
            case 2
            
                [rebound_spk(S,:),all_reb_spk(S,:)] = ...
                    TC_model_CX_SNr_cond_changed_parfor_opt(...
                                    F_SNr,0,G_SNr(S),...
                                    T,mov_onset,jit_val(S),...
                                    num_trials,dir_name_trace,S,FG_SNR);
            
            
        end

        disp(['job ', num2str(job_id), ' has finished'])
    %     end
    end
    %delete(ppm);
    dir_name_cp = [dir_name,'/res-for-colorplot/'];

    if exist(dir_name_cp,'dir') ~= 7
        mkdir(dir_name_cp)
    end

    save([dir_name_cp 'corr-' num2str(jit_val*100) '-freq_inc-' num2str(freq_increase) '-nSNr_inc-' num2str(nr_neurons_increase) '-job-' num2str(job_id)],...
        'rebound_spk','all_reb_spk','G_SNr',...
        'num_trials','NT_GS_JV_TF','F_SNr','FG_SNR')
    
    
    
end
% exit
