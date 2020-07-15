# -*- coding: utf-8 -*-

"""
Script made to calculate the average pause needed for a rebound spike to happen
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.io as sio
import numpy as np
import os
import warnings
from scipy import signal
from natsort import natsorted
from scipy.signal import find_peaks



def progressbar(acc, total, total_bar):
    """ simple progressbar """
    frac = acc/total
    filled_progbar = round(frac*total_bar)
    print('\r', '#'*filled_progbar + '-'*(total_bar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')





def neural_data_obtainer(trial_data):
    """
    obtains sorted spikes in ms, spikes in bits and vth values
    """
    all_spiketrains = trial_data[1].flatten()
    vth = trial_data[10]
    
    all_spikes_grouped = []
    spike_count = 0
    spike_bits = np.zeros((30, 100001))

    
    for i in range(all_spiketrains.size):
        data = all_spiketrains[i][0] #had to do this because of data bugs, contains group of spiketrains
        all_spikes_grouped.append(data.flatten())
        neurons, spikes = data.shape
        for neu in range(neurons):
            #create binary representation of spikes
            spike_bits[spike_count,(data[neu]*100).astype(np.int)] = 1 #transform positions to 1s
            spike_count += 1        
        
        
    all_spikes = all_spikes_grouped[0]
    for skptr in range(1,len(all_spikes_grouped)):
        all_spikes = np.concatenate((all_spikes,all_spikes_grouped[skptr]), axis=None )  
    
    
    
    return  np.sort(all_spikes), spike_bits,  vth.flatten()






def obtain_pause(all_spikes, peak):
    
    left_pause_ms = np.searchsorted(all_spikes, peak, side='right') - 1
    
    if left_pause_ms+1 == len(spikes): #case where last pause is last spike possible
        pause = 1000   -  all_spikes[left_pause_ms]
    else:
        pause = all_spikes[left_pause_ms+1]   -  all_spikes[left_pause_ms]

    return pause








pwd = os.getcwd()
warnings.filterwarnings("ignore") #due to matplotlib depreciation warnings

nr_folders = int(input("please enter number of folders with vth data: "))
nr_trials = int(input("please enter total number of trials per experiment: "))
nr_experiments = int(input("please enter total number of experiments: "))

root_exp_folders = []
corrs = [30, 40, 50, 60, 70, 80, 90, 100]
T = 100000
exp_count = 0
all_pauses = [] #here we will store the values of the pauses

for i in range(nr_folders):
    folder_results = input("please enter vth results directory # " + str(i+1) + " :")
    main_basic_exp_folder = os.path.join(pwd, folder_results) #get main experiment folder
    root_exp_folders.append(main_basic_exp_folder)
    
    


for root_folder in root_exp_folders: #root folders
    
    for i, experiment_name in enumerate(os.listdir(root_folder)): 
        freq_folder = os.path.join(root_folder, experiment_name) #freq inc experiment
        
        for j,nr_neurons_increased in enumerate(natsorted(os.listdir(os.path.join(root_folder, experiment_name)))):
                neuron_nr_folder = os.path.join(freq_folder, nr_neurons_increased) # neuron experiment
                # we now need to loop over the different correlation values
                
                for corr in corrs:
                    neural_data = sio.loadmat(neuron_nr_folder + '\\voltage-traces-and-inputs\\' + str(corr) + '-70')["mem_v_traces"]
                    spiketrains = neural_data[0] # grab the main structure mem_v_traces
                    
                    for trial in range(nr_trials):

                        
                        spiketrainsv2 = spiketrains[trial-1] #each index here is a trial
                        spikes, spike_bits, vth_vals = neural_data_obtainer(spiketrainsv2)
                       
                        
                        
                        peaks, _ = find_peaks(vth_vals[200:T], height=-40)
                        #instances (with 10 ms dedduction) when the spikes happened
                        spike_instances = (peaks / 100) - 10
                        
                        pauses = [obtain_pause(spikes, x) for x in spike_instances ]
                        all_pauses.extend(pauses)
                
                   
                    progressbar(exp_count+1, nr_experiments, 50)
                    exp_count += 1
                
                
                
                
all_pauses = np.asarray(all_pauses, dtype=np.float32)                
      
all_pauses = np.sort(all_pauses)   
np.set_printoptions(threshold=np.inf)       
print(all_pauses)                
print(np.mean(all_pauses))
print(np.std(all_pauses))
                
                
                
                
                
                
                