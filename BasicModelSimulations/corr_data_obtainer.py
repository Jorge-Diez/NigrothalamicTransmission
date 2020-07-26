# -*- coding: utf-8 -*-

"""
Script made to calculate and plot how correlation changes based on the different method that is used
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.io as sio
import numpy as np
import os
import pickle
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
    
    all_spikes_grouped = []
    spike_count = 0
    spike_bits = np.zeros((30, 100001))

    
    for i in range(all_spiketrains.size):
        data = all_spiketrains[i][0] #had to do this because of data bugs, contains group of spiketrains
        all_spikes_grouped.append(data.flatten())
        neurons, spikes = data.shape
        if (neurons > 30): #done because with diffmoth spikes with 1 spiketrain have this bug
            neurons, spikes = spikes, neurons
        
        for neu in range(neurons):
            #create binary representation of spikes
            spike_bits[spike_count,(data[neu]*100).astype(np.int)] = 1 #transform positions to 1s
            spike_count += 1        
        
        
    all_spikes = all_spikes_grouped[0]
    for skptr in range(1,len(all_spikes_grouped)):
        all_spikes = np.concatenate((all_spikes,all_spikes_grouped[skptr]), axis=None )  
    
    
    
    return  np.sort(all_spikes), spike_bits




def corr_mean_obtainer_simple(spike_bits, nr_neurons, corr_bin):
    """
    Returns the means for the two different groups of neurons in spike_bits
    This version will only return the full mean
    """
    
    jumps = int(100000 / len(np.arange(0,1000,corr_bin)))  #used later  
    spike_bits_binned = np.zeros( (30, len(np.arange(0,1000,corr_bin))))
    
    for i in range(len(np.arange(0,1000,corr_bin))-1):
        all_spike_bits_slice = spike_bits[:,i*jumps:(i+1)*jumps]
        sliced_spike_bits = np.sum(all_spike_bits_slice, axis=1)
        
        spike_bits_binned[: ,i ] = sliced_spike_bits
        
    
    #################################################################################   
    # FULL MEAN
        
    #since time goes from 0-1000 (1000 included, we add the last row to the last bin)
    i += 1
    all_spike_bits_slice = spike_bits[:,i*jumps:spike_bits.shape[1]-1]
    sliced_spike_bits = np.sum(all_spike_bits_slice, axis=1)
    spike_bits_binned[: ,i ] = sliced_spike_bits
    
    
    corr_matrix = np.corrcoef(spike_bits_binned)
    upper_mask =  np.tri(corr_matrix.shape[0], k=-1)
    upper_mask[np.diag_indices(upper_mask.shape[0])] = 1
    
    
    upper_corr_matrix = np.ma.array(corr_matrix, mask=upper_mask)
    
    #following calculations made for means
    zero_mask = np.ma.array(corr_matrix, mask=upper_mask, fill_value=0 )
    masked_zeroed_corr_matrix = zero_mask.filled()
    #calculate means
    full_mean= np.sum(masked_zeroed_corr_matrix)/ np.count_nonzero(masked_zeroed_corr_matrix)
    
    
    


    return full_mean

def corr_mean_obtainer_full(spike_bits, nr_neurons, corr_bin):
    """
    Returns the means for the two different groups of neurons in spike_bits
    This version returns the means for all groups
    """
    
    jumps = int(100000 / len(np.arange(0,1000,corr_bin)))  #used later  
    spike_bits_binned = np.zeros( (30, len(np.arange(0,1000,corr_bin))))
    
    for i in range(len(np.arange(0,1000,corr_bin))-1):
        all_spike_bits_slice = spike_bits[:,i*jumps:(i+1)*jumps]
        sliced_spike_bits = np.sum(all_spike_bits_slice, axis=1)
        
        spike_bits_binned[: ,i ] = sliced_spike_bits
        
    
    #################################################################################   
    # FULL MEAN
        
    #since time goes from 0-1000 (1000 included, we add the last row to the last bin)
    i += 1
    all_spike_bits_slice = spike_bits[:,i*jumps:spike_bits.shape[1]-1]
    sliced_spike_bits = np.sum(all_spike_bits_slice, axis=1)
    spike_bits_binned[: ,i ] = sliced_spike_bits
    
    
    corr_matrix = np.corrcoef(spike_bits_binned)
    upper_mask =  np.tri(corr_matrix.shape[0], k=-1)
    upper_mask[np.diag_indices(upper_mask.shape[0])] = 1
    
    
    upper_corr_matrix = np.ma.array(corr_matrix, mask=upper_mask)
    
    #following calculations made for means
    zero_mask = np.ma.array(corr_matrix, mask=upper_mask, fill_value=0 )
    masked_zeroed_corr_matrix = zero_mask.filled()
    #calculate means
    full_mean= np.sum(masked_zeroed_corr_matrix)/ np.count_nonzero(masked_zeroed_corr_matrix)
    
    
    

    
    #################################################################################   
    # MEAN BETWEEN THE TWO DIFFERENT GROUPS

    masked_zeroed_corr_matrix = masked_zeroed_corr_matrix[0:30-nr_neurons, 30-nr_neurons:30]
    both_groups_mean= np.sum(masked_zeroed_corr_matrix)/ np.count_nonzero(masked_zeroed_corr_matrix)



    #################################################################################   
    # MEAN BASELINE


    corr_matrix = np.corrcoef(spike_bits_binned[0:30-nr_neurons])
    upper_mask =  np.tri(corr_matrix.shape[0], k=-1)
    upper_mask[np.diag_indices(upper_mask.shape[0])] = 1
    upper_corr_matrix = np.ma.array(corr_matrix, mask=upper_mask)
    
    #following calculations made for means
    zero_mask = np.ma.array(corr_matrix, mask=upper_mask, fill_value=0 )
    masked_zeroed_corr_matrix = zero_mask.filled()
    #calculate means
    baseline_group_mean= np.sum(masked_zeroed_corr_matrix)/ np.count_nonzero(masked_zeroed_corr_matrix)






    #################################################################################   
    # MEAN FREQ INC


    corr_matrix = np.corrcoef(spike_bits_binned[30-nr_neurons:30])
    upper_mask =  np.tri(corr_matrix.shape[0], k=-1)
    upper_mask[np.diag_indices(upper_mask.shape[0])] = 1
    upper_corr_matrix = np.ma.array(corr_matrix, mask=upper_mask)
    
    #following calculations made for means
    zero_mask = np.ma.array(corr_matrix, mask=upper_mask, fill_value=0 )
    masked_zeroed_corr_matrix = zero_mask.filled()
    #calculate means
    freqinc_group_mean= np.sum(masked_zeroed_corr_matrix)/ np.count_nonzero(masked_zeroed_corr_matrix)
    





    return full_mean, both_groups_mean, baseline_group_mean, freqinc_group_mean








pwd = os.getcwd()
warnings.filterwarnings("ignore") #due to matplotlib depreciation warnings

root_folder = input("please enter name of the root folder: ")
nr_trials = int(input("please enter total number of trials per experiment: "))
nr_experiments = int(input("please enter total number of experiments: "))
corrbin =  float(input("select bin in ms for correlation matrix : "))


root_exp_folders = []
corrs = [30, 40, 50, 60, 70, 80, 90, 100]
T = 100000
exp_count = 0
all_pauses = [] #here we will store the values of the pauses



#f = open("corr_results_" + root_folder +".txt", "w+")
#f_corr = open("all_trials_corr_results_" + root_folder +".txt", "w+")    
#f_trialsTQ = open("all_trials_TQ_results_" + root_folder +".txt", "w+")  




ALL_CORR_values_30 = np.zeros((30,40))
ALL_CORR_values_40 = np.zeros((30,40))
ALL_CORR_values_50 = np.zeros((30,40))
ALL_CORR_values_60 = np.zeros((30,40))
ALL_CORR_values_70 = np.zeros((30,40))
ALL_CORR_values_80 = np.zeros((30,40))
ALL_CORR_values_90 = np.zeros((30,40))
ALL_CORR_values_100 = np.zeros((30,40))



ALL_CORR_differences_30 = np.zeros((30,40))
ALL_CORR_differences_40 = np.zeros((30,40))
ALL_CORR_differences_50 = np.zeros((30,40))
ALL_CORR_differences_60 = np.zeros((30,40))
ALL_CORR_differences_70 = np.zeros((30,40))
ALL_CORR_differences_80 = np.zeros((30,40))
ALL_CORR_differences_90 = np.zeros((30,40))
ALL_CORR_differences_100 = np.zeros((30,40))



ALL_CORR_percentages_30 = np.zeros((30,40))
ALL_CORR_percentages_40 = np.zeros((30,40))
ALL_CORR_percentages_50 = np.zeros((30,40))
ALL_CORR_percentages_60 = np.zeros((30,40))
ALL_CORR_percentages_70 = np.zeros((30,40))
ALL_CORR_percentages_80 = np.zeros((30,40))
ALL_CORR_percentages_90 = np.zeros((30,40))
ALL_CORR_percentages_100 = np.zeros((30,40))




ALL_CORR_MAT_LIST = [ALL_CORR_values_30, ALL_CORR_values_40, ALL_CORR_values_50, 
                    ALL_CORR_values_60, ALL_CORR_values_70, ALL_CORR_values_80,
                    ALL_CORR_values_90, ALL_CORR_values_100]




ALL_CORRDIFF_MAT_LIST = [ALL_CORR_differences_30, ALL_CORR_differences_40, ALL_CORR_percentages_50, 
                    ALL_CORR_differences_60, ALL_CORR_differences_70, ALL_CORR_differences_80,
                    ALL_CORR_differences_90, ALL_CORR_differences_100]



ALL_CORRPERC_MAT_LIST = [ALL_CORR_percentages_30, ALL_CORR_percentages_40, ALL_CORR_differences_50, 
                    ALL_CORR_percentages_60, ALL_CORR_percentages_70, ALL_CORR_percentages_80,
                    ALL_CORR_percentages_90, ALL_CORR_percentages_100]




   
 
for i, experiment_name in enumerate(os.listdir(root_folder)): 
    freq_folder = os.path.join(root_folder, experiment_name) #freq inc experiment
    
    for j,nr_neurons_increased in enumerate(natsorted(os.listdir(os.path.join(root_folder, experiment_name)))):
            neuron_nr_folder = os.path.join(freq_folder, nr_neurons_increased) # neuron experiment
            # we now need to loop over the different correlation values
            
            for c,corr in enumerate(corrs):
                neural_data = sio.loadmat(neuron_nr_folder + '\\voltage-traces-and-inputs\\' + str(corr) + '-70')["mem_v_traces"]
                spiketrains = neural_data[0] # grab the main structure mem_v_traces
                full_means = np.zeros(nr_trials)

                
                for trial in range(nr_trials):

                    
                    spiketrainsv2 = spiketrains[trial-1] #each index here is a trial
                    spikes, spike_bits = neural_data_obtainer(spiketrainsv2)
                   
                    fmean = corr_mean_obtainer_simple(spike_bits, j, corrbin)
                    full_means[trial] = fmean
                    
                    
                    
                    
                    
               
                full_mean_corr = np.mean(full_means)
                #mat_data = sio.loadmat(neuron_nr_folder + '\\plotting-params-MIP')
                #tq = (mat_data["OVR"]).flatten()
                #tq_corr = tq[c]
                
                #f.write("Freq{}_neurons{}_corr{}: {}\n".format(str(i+51), j+1, corr, str(full_mean_corr)))
                #f_corr.write("{}\n".format(full_mean_corr))
                #f_trialsTQ.write("{}\n".format(tq_corr))
                
                
                
                
                
                
                diff_from_intented_corr = (corr/100) - full_mean_corr
                percentage_variation = diff_from_intented_corr / (corr/100) * 100
                
                CORR_MATRIX = ALL_CORR_MAT_LIST[c]
                CORRDIFF_MATRIX = ALL_CORRDIFF_MAT_LIST[c]
                CORRPERC_MATRIX = ALL_CORRPERC_MAT_LIST[c]

                
                
                CORR_MATRIX[j][i] = full_mean_corr
                CORRDIFF_MATRIX[j][i] = diff_from_intented_corr
                CORRPERC_MATRIX[j][i] = percentage_variation
                
                
                
                
                progressbar(exp_count+1, nr_experiments, 50)
                exp_count += 1
                
                
                
                
#PICKLE AND SAVE THE RESULTS SINCE THEY TAKE QUITE SOME TIME TO GENERATE
                

with open('corrmatrixes' +  root_folder + '.pickle', 'wb') as handle:
        pickle.dump(ALL_CORR_MAT_LIST, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
with open('corrdiffmatrixes' +  root_folder + '.pickle', 'wb') as handle:
        pickle.dump(ALL_CORRDIFF_MAT_LIST, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
with open('corrpercmatrixes' +  root_folder + '.pickle', 'wb') as handle:
        pickle.dump(ALL_CORRPERC_MAT_LIST, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                
                
#f.close()       
#f_corr.close()         
#f_trialsTQ.close()
                
                