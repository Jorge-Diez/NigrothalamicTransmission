%% TC_model_CX_SNr_cond_changed.m description
% This function is written to simulate the neuron with Poissonian inputs
% both from CX and SNr

% the function is written in a way that some input parameters can be changed.
% Also the output of this function is a binary which tells us whether
% rebound happened and count the number of spike. The algorithm is based on
% minimum value of Ica current. The number of spikes are check in a
% predefined window of size 50 ms which should be corrected.

%% added on 01.07.15

% All other event-unrelated rebound spikes before the event is also
% outputed.

%% Function definition

function [MOV_REB,ALL_REB] = ...
    TC_model_CX_SNr_cond_changed_parfor_opt...
                                          (...
                                          f_snr,...
                                          corr_snr,...
                                          g_snr,...
                                          T,...
                                          mov_onset,...
                                          deg_of_jit,...
                                          num_trials,dir_name_trace,S,FG_SNR)

%% Global variables

%global dir_name

%% input parameters determination

% switch nargin
%     case 2
%         N_CX = n_cx;
%         N_SNr = n_snr;
%         F_CX = 2;
%         F_SNr = 2;
%         corr_CX = 0;
%         corr_SNr = 0;
%     case 4
%         N_CX = n_cx;
%         N_SNr = n_snr;
%         F_CX = f_cx;
%         F_SNr = f_snr;
%         corr_CX = 0;
%         corr_SNr = 0;
%     case 6
%         N_CX = n_cx;
        %N_SNr = n_snr;
%         F_CX = f_cx*ones(size(T));
        F_SNr = f_snr;%*sigmf(T,[-0.1 mov_onset]);
%         corr_CX = corr_cx;
        corr_SNr = corr_snr;
% end

% disp([corr_snr,g_snr])

%% Simulation parameters

stepsize = T(2)-T(1);
simtime = T(end);

%% TC parameters

%seen in simulink model
asg = 200;
bsg = 0.4;
itc = 6;
shi = -80;
shi2 = -90;
dur = 5;
dur2 = 10;
period = 25;
gnabar = 3;
gkbar = 5;
glbar = 0.05;
ena = 50;
ek = -90;
eleak = -70;
gtbar = 5;
qht = 5.5;
tadj = 1;
apr = 4;
apt = 0.3;

%% Initial Conditions

vt0 = 0;
ht0 = 0;
htt0 = 0;

%% Synaptic parameters (SNr to TH)

vsynSNrtoTH = -85;
gsynSNrtoTH = g_snr;    % It is multiplied by 8 because in the origiran model
                    % that is used to investigate the effect of DBS, 8 GPi
                    % neurons provide inputs for a single TC cell
alphaSNrtoTH = 1;
betaSNrtoTH = 0.08;
s0SNrtoTH = 0;

%% Synaptic parameters (SNr to TH)

vsynCXtoTH = 0;
gsynCXtoTH = 0.05;    % It is multiplied by 8 because in the origiran model
                    % that is used to investigate the effect of DBS, 8 GPi
                    % neurons provide inputs for a single TC cell
alphaCXtoTH = 1.1;
betaCXtoTH = 0.19;
s0CXtoTH = 0;

%% Constant input to the model
% 
% constant_input = 0;
% 
% step_time = 500;
% %simtime = 2000;
% 
% period = simtime/2;
% pulse_width = 20;
% phase_delay = 500;

%% Poissonian input to the TC model

% input from SNr
% mov_onset = simtime/2;
mem_v_traces = struct([]);
MOV_REB = zeros(1,num_trials);
ALL_REB = zeros(1,num_trials);
R2_vec = zeros(1,num_trials);


%start of simulation of model neuron for each trial
for tr_ind = 1:num_trials
    %constrain random number generator of matlab to give a certain spike train for each trial
    rng(tr_ind)
    
    
    %disp(['S = ',num2str(S)])

    %specify random generator
    
    %spikes that will be used in simulink, need to be reshaped into a
    %single array
    jit_spk_times = [];
    %struct which will be saved into exported data
    all_spikes = struct();
    
    %to try and see why cluster gives back different results 
    mem_v_traces(tr_ind).rng = rng;
    %generate spikes with highest firing rate and all N
    [spike_times] = MIP_imp_v4_beta(deg_of_jit,sum(FG_SNR),max(F_SNr),T(T<=mov_onset));
    
    
    
    %THIS IS DONE TAKING INTO ACCOUNT THAT TIME VECTOR IS 1 SECOND = 1000
    %MILLISECONDS AND THAT ONLY 2 GROUPS OF NEURONS EXIST (BASELINE AND
    %HIGHER FREQUENCY)
    if (numel(F_SNr) > 1)
        base_spikes = zeros(FG_SNR(1), F_SNr(1));
        %go over each neuron and delete randomly
        base_freq = F_SNr(1);
        for i = 1:FG_SNR(1)
            %we will keep only the number of spikes that we need
            positions_keep = randperm(F_SNr(2), base_freq);
            base_spikes(i,:) = spike_times(i, positions_keep);
        end
        
        %save first "group" of base freq
        all_spikes(1).spikes = base_spikes;
        jit_spk_times = vertcat( jit_spk_times, (reshape(base_spikes,numel(base_spikes),1)));
        
        
        %save neurons with higher frequency 
        stimulated_spikes = spike_times(FG_SNR(1)+1:sum(FG_SNR),:);
        all_spikes(2).spikes = stimulated_spikes;
        %transform spikes for simulink
        jit_spk_times = vertcat( jit_spk_times, (reshape(stimulated_spikes,numel(stimulated_spikes),1)));        
    else
        
        all_spikes(1).spikes = spike_times;
        jit_spk_times = (reshape(spike_times,numel(spike_times),1));
        
    
    end
    
    
    %% Showing the results
    simopt = simset('SrcWorkspace','current');  %
    sim('TC_RT2004_pois_CX_SNr',[],simopt) %running the simulink (similar to run)

    % dir_name = [pwd '/' num2str(N_SNr) '/ACTIVITY'];
    % 
    % if exist(dir_name,'dir') ~= 7 
    %     mkdir(dir_name)
    % end
    % 
    % save([dir_name,'/',...
    %     num2str(trial_num),...
    %     '-',num2str(round(deg_of_jit*100)),...
    %     '-',num2str(round(g_snr*100))],...
    %    'I_SNrtoTH','Ica','vth','mov_onset')
    
    %structure to save important outputs
        mem_v_traces(tr_ind).mov_onset = mov_onset; %movement onset happend
        mem_v_traces(tr_ind).spike_times = all_spikes;
        mem_v_traces(tr_ind).des_corr = deg_of_jit; %correlation
%         mem_v_traces(tr_ind).exp_num = exp_num;


    % save([pwd '/' dir_name '/ACTIVITY/' num2str(round(corr_SNr*100)) '-' num2str(round(g_snr*100))],...
    %    'I_SNrtoTH','Ica','vth','mov_onset')

    %close(gcf) 

    %% Rebound spike detection

    % [PKS,LOCs] = findpeaks(diff(Ica.signals.values(Ica.time>=mov_onset)),...
    %     'MINPEAKHEIGHT',0.05);
    mov_rebound_spk = 0;
    
    %% The detailed detection algorithm

%     % Finding the rebound spikes before movement related spike
% 
%     all_rebound_spk = length(findpeaks(abs(Ica.signals.values(Ica.time<mov_onset)),... 
%                             'MINPEAKHEIGHT',1.861));
%     Ica_mov = Ica.signals.values(Ica.time>=mov_onset);
%     vth_mov = vth.signals.values(vth.time>=mov_onset);
%     t_mov = Ica.time(Ica.time>=mov_onset);
%     min_Ica = min(Ica_mov);
%     if min_Ica <= -1.861
%         try
%             [pks,temp_ind] = findpeaks(abs(Ica_mov),'MINPEAKHEIGHT',1.861);
%             rebound_ind = temp_ind(1);
%             st_time = t_mov(rebound_ind);
%             end_time = st_time + 50;
%             end_rebound_ind = find(t_mov <= end_time);
%             end_rebound_ind = end_rebound_ind(end);
% 
%             if ~isempty(rebound_ind) && ~isempty(end_rebound_ind)
%                 mov_rebound_spk = length(findpeaks(vth_mov(rebound_ind:end_rebound_ind),...
%                     'MINPEAKHEIGHT',-40));
%             end
% 
%         catch
%             mov_rebound_spk = 0;
%         end
%     end

    %% The simpler algorithm for spike detection
    warning off
    all_rebound_spk = length(findpeaks(vth.signals.values(vth.time<mov_onset),... 
                            'MINPEAKHEIGHT',-40)); %count how many spikes before movement onset
                        
    mov_rebound_spk = length(findpeaks(vth.signals.values(vth.time>=mov_onset),... 
                            'MINPEAKHEIGHT',-40)); % count how many spikes after movement onset
    
    MOV_REB(tr_ind) = mov_rebound_spk;
    ALL_REB(tr_ind) = all_rebound_spk;
%     R2_vec(tr_ind) = R2;
    mem_v_traces(tr_ind).corr = deg_of_jit;
    mem_v_traces(tr_ind).mov_reb = mov_rebound_spk;
    mem_v_traces(tr_ind).all_reb = all_rebound_spk;
    mem_v_traces(tr_ind).g_snr = g_snr;
    mem_v_traces(tr_ind).mode = "spike_deletion";
    mem_v_traces(tr_ind).F_SNr = F_SNr;
    mem_v_traces(tr_ind).FG_SNR = FG_SNR;
    mem_v_traces(tr_ind).vthvalues = vth.signals.values;

    
    disp(['trial ', num2str(tr_ind), ' has finished'])
end

save([dir_name_trace,num2str(round(deg_of_jit*100)),'-',...
    num2str(round(g_snr*100))],...
   'mem_v_traces')



