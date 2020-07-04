# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import os
import warnings



def progressbar(acc, total, total_bar):
    """ simple progressbar """
    frac = acc/total
    filled_progbar = round(frac*total_bar)
    print('\r', '#'*filled_progbar + '-'*(total_bar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')





pwd = os.getcwd()
warnings.filterwarnings("ignore") #due to matplotlib depreciation warnings


folder_results = input("please enter the directory where the results are located:")
main_basic_exp_folder = os.path.join(pwd, folder_results) #get main experiment folder
corr = input("please indicate correlation: ")
neural_data = sio.loadmat(main_basic_exp_folder + '\\voltage-traces-and-inputs\\' + corr + '-70')["mem_v_traces"]
RESULTS_FOLDER = "SPIKETRAIN_GRAPHIC"
if not os.path.exists(RESULTS_FOLDER):
    os.mkdir(RESULTS_FOLDER)
exp_name = input("please enter name for experiment (sub-folder and plot titles will have this): ")
RESULTS_FOLDER = "SPIKETRAIN_GRAPHIC\\" + exp_name
if not os.path.exists(RESULTS_FOLDER):
    os.mkdir(RESULTS_FOLDER)



spiketrains = neural_data[0] # grab the main structure mem_v_traces
spiketrainsv2 = spiketrains[0]
all_spiketrains = spiketrainsv2[1].flatten()


#groups_spikes = int(input("Please indicate numer of groups of neurons with different firing rates: "))

spike_count = 0
all_spikes_grouped = []
tot_amount_spikes = 0

# create numpy array for 1s and 0s
# we are going to suppose we have a max of 30 spikes
spike_bits = np.zeros((30, 100001))



plt.figure(1, figsize=(19,10))
for i in range(all_spiketrains.size):
    data = all_spiketrains[i][0] #had to do this because of data bugs, contains group of spiketrains
    neurons, spikes = data.shape
    all_spikes_grouped.append(data.flatten())
    tot_amount_spikes += (data.flatten()).size
    for neu in range(neurons):
        #draw the vertical lines
        plt.plot([data[neu],data[neu]],[spike_count,spike_count+1], 'b')
        
        #create binary representation of spikes
        spike_bits[spike_count,(data[neu]*100).astype(np.int)] = 1 #transform positions to 1s
        
        
        spike_count += 1
        

plt.title('Spike raster plot for ' + exp_name)
plt.xlabel('Time (ms)')
plt.ylabel('Neurons')
plt.savefig(RESULTS_FOLDER + "\\" + "Spiketrains",bbox_inches='tight')









all_spikes = all_spikes_grouped[0]
for skptr in range(1,len(all_spikes_grouped)):
      all_spikes = np.concatenate((all_spikes,all_spikes_grouped[skptr]), axis=None )  
    
all_spikes = np.sort(all_spikes)
minpause = float(input("select min pause in ms to show in rasterplot: "))
total_pause_count = 0 #coutns how many ms of total pauses
for i in range (len(all_spikes)-1):
     
    if (all_spikes[i+1] - all_spikes[i] >= minpause):
        #plot the vertical lines between the pause points
        plt.plot( [all_spikes[i], all_spikes[i]], [0, spike_count  ], 'r--')
        plt.plot( [all_spikes[i+1], all_spikes[i+1]], [0, spike_count  ], 'r--')
        
        #plot horizontal lines at the top and bottom
        plt.plot( [all_spikes[i], all_spikes[i+1]], [0, 0  ], 'r')
        plt.plot( [all_spikes[i], all_spikes[i+1]], [spike_count, spike_count  ], 'r')
        
        #add pause to total
        total_pause_count += all_spikes[i+1] - all_spikes[i]
    

#add graph information
plt.title('Spike raster plot with synchronous pauses of >= ' + str(minpause) + 'ms for  ' + exp_name + "\n"
          + "Total time in ms paused: " + str(total_pause_count) )

plt.xlabel('Time (ms)')
plt.ylabel('Neurons')
plt.savefig(RESULTS_FOLDER + "\\" + "Spiketrains_sync_pauses",bbox_inches='tight')







############################################################################################################
############################################################################################################


#amplitude distribution
# 0.01 ms bin 
t_vec = np.arange(0,1000.01,0.01)
hist, bins = np.histogram(all_spikes, t_vec)




plt.figure(2, figsize=(10,8))
plt.hist(hist[hist != 0], range(31), ec='black')
locs, labels = plt.yticks()
plt.title("Amplitude distribution for " + exp_name)
plt.xlabel('Amplitude')
plt.ylabel('Nr of repetitions')
plt.savefig(RESULTS_FOLDER + "\\" + "Amplitude distribution",bbox_inches='tight')






############################################################################################################
############################################################################################################


bars, bars_counts = np.unique(np.sort(hist), return_counts = True)
#delete cases for 0
bars_full = np.zeros(31)

bars = np.delete(bars, 0)
bars_counts = np.delete(bars_counts, 0)
bars_full[bars] = bars_counts

probs_full = bars_full/bars_full.sum()

plt.figure(3, figsize=(10,8))
plt.title("Amplitude-Frequency distribution for " + exp_name )

plt.bar(np.arange(len(bars_full)),probs_full)
plt.xlabel('Amplitude')
plt.ylabel('Probability')
plt.savefig(RESULTS_FOLDER + "\\" + "Amplitude-Frequency distribution",bbox_inches='tight')

#plt.xticks(np.arange(len(bars_full)), bars_full)



""" 






orig_uniques, orig_counts = np.unique(all_spikes, return_counts = True)
bars, bars_counts = np.unique(np.sort(orig_counts), return_counts = True)
probs = bars_counts/bars_counts.sum()

plt.figure(2)
plt.title("MIP with unique done directly")
plt.bar(np.arange(len(bars)),probs)
plt.xticks(np.arange(len(bars)), bars)
"""

############################################################################################################
############################################################################################################

np.sort(all_spikes)
bin = float(input("select bin for amplitude-probability distribution: "))
counts = []
for i in np.arange(0,1000,bin):
    counts.append(len(list(x for x in all_spikes if i <= x <= i+bin)))
    

bars_bins, bar_bins_counts = np.unique(np.sort(counts), return_counts = True)
#delete cases for 0
bars_bins = np.delete(bars_bins, 0)
bar_bins_counts = np.delete(bar_bins_counts, 0)
probs_bins = bar_bins_counts/bar_bins_counts.sum()
plt.figure(4, figsize=(10,8))
plt.title("Amplitude distribution for bins of " + str(bin) + " ms for " + exp_name)
plt.bar(np.arange(len(bars_bins)),probs_bins)
plt.xlabel('Amplitude')
plt.ylabel('Probability')
plt.xticks(np.arange(len(bars_bins)), bars_bins)
plt.savefig(RESULTS_FOLDER + "\\" + "Amplitude_distribution_bins",bbox_inches='tight')




############################################################################################################
############################################################################################################
#inverting spike binary represantion to show pauses


inverted_spike_bits = 1 - spike_bits
#changing bits to actual times when there was no firing (inverse of firing)
all_inverted_spikes = (np.nonzero(inverted_spike_bits[0,:])[0])/100
for skptr in range(1,len(inverted_spike_bits)):
      all_inverted_spikes = np.concatenate((all_inverted_spikes,
                                            (np.nonzero(inverted_spike_bits[skptr,:])[0])/100), axis=None )



np.sort(all_inverted_spikes)
t_vec = np.arange(0,1000.01,0.01)
hist, bins = np.histogram(all_inverted_spikes, t_vec)




plt.figure(5, figsize=(10,8))
plt.hist(hist[hist != 0], range(30), ec='black')
locs, labels = plt.yticks()
plt.title("Amplitude distribution of pauses for " + exp_name)
plt.xlabel('Amplitude')
plt.ylabel('Nr of repetitions')
plt.savefig(RESULTS_FOLDER + "\\" + "Amplitude distribution pauses without n = 30",bbox_inches='tight')




############################################################################################################
############################################################################################################


bin = float(input("select bin for amplitude-probability distribution of pauses: "))
counts = []
print("Calculating amplitude-probability distibution for pauses.....")
iterations = len(np.arange(0,1000,bin))
for progress,i in enumerate(np.arange(0,1000,bin)):
    counts.append(len(list(x for x in all_inverted_spikes if i <= x <= i+bin)))
    progressbar(progress+1, iterations, 50)
    

bars_bins, bar_bins_counts = np.unique(np.sort(counts), return_counts = True)
#delete cases for 0
bars_bins = np.delete(bars_bins, 0)
bar_bins_counts = np.delete(bar_bins_counts, 0)
probs_bins = bar_bins_counts/bar_bins_counts.sum()
plt.figure(6, figsize=(19,10))
plt.title("Amplitude distribution of pauses for bins of " + str(bin) + " ms for " + exp_name)
plt.xlabel('Amplitude')
plt.ylabel('Probability')
plt.bar(np.arange(len(bars_bins)),probs_bins)
plt.xticks(np.arange(len(bars_bins)), bars_bins, fontsize=7)
plt.savefig(RESULTS_FOLDER + "\\" + "Amplitude_distribution_bins_pauses",bbox_inches='tight')








############################################################################################################
############################################################################################################





corr_matrix = np.cov(spike_bits)
corr_matrix_upper = np.triu(corr_matrix)




plt.show()












