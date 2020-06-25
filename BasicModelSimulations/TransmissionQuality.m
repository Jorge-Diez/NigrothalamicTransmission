%% Date created 15.08.18 by M. Mohagheghi

% This script generates the results for the transmission quality for
% exponential and binomial amplitude distributions

addpath('functions')

% Exponential

% Running the simulation; 1st input arg. specifies which chunk in
% parameter space should be run and 2nd arg. specifies how many chunks
% should the parameter space be devided into

%avoid all warnings
warning('off','all')
warning

mov_onset = 1000; % in ms
N_CX = 200; %Do not contemplate for now

F_CX = 1:0.5:10; %Do not contemplate for now

sim_mode = 1; % 1 is different mother spikes, 2 is spike deletion method


G_SNr_all = 0.70; %nigral conductance
corr_vals = 0.3:0.1:1; %values of correlation among inhibitory inputs
num_trials = 10;


%The following are the parameters for our simulators
%We will use 2 for loops in order for us to be able to use the double
%progress bar
% the following is the increase in % with respect to base 50 HZ
min_perc_increase = 20;
max_perc_increase = 20;

percentage_increases = [min_perc_increase:2:max_perc_increase] / 100;
N_SNr = 30;

%number of neurons that will have that increase in percentage
neurons_with_increase = [1:N_SNr];

nr_perc_experiments = length(percentage_increases);
nr_neuron_experiments = length(neurons_with_increase);

%create multiwaitbar with out bars
multiWaitbar(['nr of perc experiments done : 0', ' out of ', num2str(nr_perc_experiments) ], 0);
multiWaitbar(['nr of neuron experiments done : 0', ' out of ', num2str(nr_neuron_experiments) ], 0);
total_number_experiments = nr_perc_experiments*nr_neuron_experiments;

disp(['TOTAL NUMBER OF EXPERIMENTS TO BE PERFORMED: ',num2str(total_number_experiments) ] )

experiments_performed = 0;
last = 0;

for per_i = 1:nr_perc_experiments
    for nr_j = 1:nr_neuron_experiments
        
        if (nr_j == N_SNr)
            F_SNr = [50 + (percentage_increases(per_i) * 50)]; %Hz

            F_Group_neurons = [nr_j]; %groups of neurons with specific firing rates
            
        else

        %obtain parameters
            F_SNr = [50 50 + (percentage_increases(per_i) * 50)]; %Hz

            F_Group_neurons = [N_SNr - neurons_with_increase(nr_j) nr_j]; %groups of neurons with specific firing rates
        end
        
        
        
        
        %create directory path to save results from this experiment
        root_folder = ['TOCHECKREPRODUCABILITY\FREQINCTO_',num2str(50 + (percentage_increases(per_i) * 50))];

        checkflag = fullfile(pwd, root_folder);
        if exist(checkflag,'dir') ~= 7
            mkdir(checkflag)
        end

        specific_folder = ['perc_',num2str(percentage_increases(per_i)),'_nr_neurons_',num2str(neurons_with_increase(nr_j))];
        exp_path = fullfile(root_folder,specific_folder );

        res_dir_mip = TCmodel_func_bwfor(1, 1 , mov_onset, N_CX, F_CX, F_SNr, G_SNr_all, num_trials, corr_vals,F_Group_neurons,sim_mode,exp_path);  %binomial
        vis_res_lumped_mats(res_dir_mip, 'MIP')

        
        
        %caculate percentages performed
        frac2 = nr_j / length(neurons_with_increase);
        frac1 = (per_i-1 + frac2) / length(percentage_increases);
        experiments_performed = experiments_performed+1;

        %used to update number of experiments performed
        handle = mod(experiments_performed-1,nr_neuron_experiments );
        
        %update percentage
        multiWaitbar(['nr of neuron experiments done : ',num2str(handle), ' out of ', num2str(nr_neuron_experiments) ], 'Value', frac2 );
    
        %update number of experiments performed
        multiWaitbar(['nr of neuron experiments done : ',num2str(handle), ' out of ', num2str(nr_neuron_experiments) ], 'Relabel',...
        ['nr of neuron experiments done : ',num2str(nr_j), ' out of ', num2str(nr_neuron_experiments) ]);
        
        
    end
    %resets number of neuron experiments counter
    multiWaitbar(['nr of neuron experiments done : ',num2str(N_SNr), ' out of ', num2str(nr_neuron_experiments) ], 'Relabel',...
    ['nr of neuron experiments done : ',num2str(0), ' out of ', num2str(nr_neuron_experiments) ]);  
   

    %update percentage
    multiWaitbar(['nr of perc experiments done : ',num2str(per_i-1), ' out of ', num2str(nr_perc_experiments) ], 'Value', frac1 );
     
    %update number of experiments performed
    multiWaitbar(['nr of perc experiments done : ',num2str(per_i-1), ' out of ', num2str(nr_perc_experiments) ], 'Relabel',...
    ['nr of perc experiments done : ',num2str(per_i), ' out of ', num2str(nr_perc_experiments) ]);
end
multiWaitbar( 'CloseAll' );










%disp(['Showing the summary results ...'])
%res_summary(res_dir_exp, res_dir_mip)