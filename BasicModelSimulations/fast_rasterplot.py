# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.io as sio
import numpy as np
import os
import warnings
from scipy import signal

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
trial = int(input("please enter trial number: "))

neural_data = sio.loadmat(main_basic_exp_folder + '\\voltage-traces-and-inputs\\' + corr + '-70')["mem_v_traces"]
RESULTS_FOLDER = "SPIKETRAIN_GRAPHIC"
if not os.path.exists(RESULTS_FOLDER):
    os.mkdir(RESULTS_FOLDER)
exp_name = input("please enter name for experiment (sub-folder and plot titles will have this): ")
RESULTS_FOLDER = "SPIKETRAIN_GRAPHIC\\" + exp_name
if not os.path.exists(RESULTS_FOLDER):
    os.mkdir(RESULTS_FOLDER)



spiketrains = neural_data[0] # grab the main structure mem_v_traces
spiketrainsv2 = spiketrains[trial-1] #each index here is a trial
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
    if (neurons > 30): #done because with diffmoth spikes with 1 spiketrain have this bug
            neurons, spikes = spikes, neurons
        
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
minpause = 35
total_pause_count = 0 #coutns how many ms of total pauses
ax = plt.gca()
for i in range (len(all_spikes)-1):
     
    if (all_spikes[i+1] - all_spikes[i] >= minpause):
        
        """
        #plot the vertical lines between the pause points
        plt.plot( [all_spikes[i], all_spikes[i]], [0, spike_count  ], 'r--')
        plt.plot( [all_spikes[i+1], all_spikes[i+1]], [0, spike_count  ], 'r--')
        """
        
        #plot horizontal lines at the top 
        #plt.plot( [all_spikes[i], all_spikes[i+1]], [0, 0  ], 'r', linewidth=7.0)
        rect_length = (all_spikes[i+1] - all_spikes[i]) - 2.5
        rect_pause = plt.Rectangle((all_spikes[i]+2.5, spike_count+0.5),rect_length, 0.5, color='r')
        ax.add_patch(rect_pause)
        #plt.plot( [all_spikes[i]+2, all_spikes[i+1]-2], [spike_count+0.5, spike_count+0.5  ], 'r')
        
        #add pause to total
        total_pause_count += all_spikes[i+1] - all_spikes[i]


#adding last pause (if there is such)
#simulation ends at 1000ms so we check if last spike has a significant difference with respect to that
if (  (1000 - all_spikes[len(all_spikes)-1]) >= minpause  ):
    rect_length = (1000 - all_spikes[len(all_spikes)-1]) - 2.5
    rect_pause = plt.Rectangle((all_spikes[len(all_spikes)-1]+2.5, spike_count+0.5),rect_length, 0.5, color='r')
    ax.add_patch(rect_pause)    

#add graph information
plt.title('Spike raster plot with synchronous pauses of >= ' + str(minpause) + 'ms for  ' + exp_name + "\n"
          + "Total time in ms paused: " + str(total_pause_count) )

plt.xlabel('Time (ms)')
plt.ylabel('Spike train #')
plt.savefig(RESULTS_FOLDER + "\\" + "Spiketrains_sync_pauses",bbox_inches='tight')












############################################################################################################
############################################################################################################
#AMPLITUDE DISTRIBUTION

# 0.01 ms bin 
t_vec = np.arange(0,1000.01,0.01)
hist, bins = np.histogram(all_spikes, t_vec)




plt.figure(2, figsize=(10,8))
plt.hist(hist[hist != 0], range(31), ec='black')
locs, labels = plt.yticks()
plt.title("Amplitude distribution of step size for " + exp_name)
plt.xlabel('Amplitude')
plt.ylabel('Nr of repetitions')
plt.savefig(RESULTS_FOLDER + "\\" + "Amplitude distribution",bbox_inches='tight')






############################################################################################################
############################################################################################################
#AMPLITUDE FREQUENCY DISTRIBUTION


bars, bars_counts = np.unique(np.sort(hist), return_counts = True)
#delete cases for 0
bars_full = np.zeros(31)

bars = np.delete(bars, 0)
bars_counts = np.delete(bars_counts, 0)
bars_full[bars] = bars_counts

probs_full = bars_full/bars_full.sum()

plt.figure(3, figsize=(10,8))
plt.title("Amplitude-Frequency distribution of step size for " + exp_name )

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
#BINNED AMPLITUDE DISTRIBUTION

np.sort(all_spikes)
bin = 1


t_vec = np.arange(0,1000+bin,bin)
hist, bins = np.histogram(all_spikes, t_vec)

max_bin = max(hist)


plt.figure(4, figsize=(10,8))
plt.hist(hist[hist != 0], range(max_bin+1), ec='black')
locs, labels = plt.yticks()
plt.title("Amplitude distribution for bins of " + str(bin) + " ms for " + exp_name)
plt.xlabel('Amplitude')
plt.ylabel('Counts')
plt.savefig(RESULTS_FOLDER + "\\" + "Amplitude_distribution_bins",bbox_inches='tight')



############################################################################################################
############################################################################################################
#BINNED AMPLITUDE FREQUENCY DISTRIBUTION


bars, bars_counts = np.unique(np.sort(hist), return_counts = True)
#delete cases for 0
bars_full = np.zeros(max_bin+1)

bars = np.delete(bars, 0)
bars_counts = np.delete(bars_counts, 0)
bars_full[bars] = bars_counts

probs_full = bars_full/bars_full.sum()

plt.figure(5, figsize=(10,8))
plt.title("Amplitude-Frequency distribution for bins of " + str(bin) + " ms for " + exp_name)

plt.bar(np.arange(len(bars_full)),probs_full)
plt.xlabel('Amplitude')
plt.ylabel('Probability')
plt.savefig(RESULTS_FOLDER + "\\" + "Amplitude-Frequency distribution bins",bbox_inches='tight')









############################################################################################################
############################################################################################################
#PAUSE AMPLITUDE DISTRIBUTION

#changing bits to actual times when there was no firing (inverse of firing)
inverted_spike_bits = 1 - spike_bits

all_pauses =  np.zeros(inverted_spike_bits.shape[1])

#logical and to get all pauses in a single vector
for i in range(inverted_spike_bits.shape[1]):
    if (all(inverted_spike_bits[:,i])):
        all_pauses[i] = 1


#count how many steps are in each pause
consec_pauses = []

temp_counter = 0
for i in all_pauses:
    if (i):
        temp_counter += 1
    else:
        consec_pauses.append(temp_counter)
        temp_counter = 0

#save last set
consec_pauses.append(temp_counter)       
consec_pauses = np.asarray(consec_pauses, dtype=np.float32)
consec_pauses = consec_pauses[consec_pauses != 0]

#convert pauses to time in ms, because they are pauses in 0.01ms steps
consec_pauses = consec_pauses/100



plt.figure(6, figsize=(10,8))
plt.hist(consec_pauses, 20, ec='black')
plt.title("Amplitude distribution of pauses for " + exp_name)
plt.xlabel('Pauses in ms')
plt.ylabel('Counts')
plt.savefig(RESULTS_FOLDER + "\\" + "Amplitude distribution pauses",bbox_inches='tight')








############################################################################################################
############################################################################################################
# FIRING RATES

nr_neurons = int(input("Indicate how many neurons are in group with freq. increase: "))



firing_rate_bin = 1
jumps = int(100000 / len(np.arange(0,1000,firing_rate_bin)))  #used later  

all_firing_rate = []
baseline_firing_rate = []
freqinc_firing_rate = []

#im aware i can add baseline and freqinc im just incredibly paranoid from not sleeping and i dont trust anything 
for i in range(len(np.arange(0,1000,firing_rate_bin))-1):
    all_spike_bits_slice = spike_bits[:,i*jumps:(i+1)*jumps]
    baseline_spike_bits_slice = spike_bits[0:30-nr_neurons,i*jumps:(i+1)*jumps]
    freqinc_spike_bits_slice = spike_bits[30-nr_neurons:30,i*jumps:(i+1)*jumps]
    
    
    all_nr_spikes = np.count_nonzero(all_spike_bits_slice)
    baseline_nr_spikes = np.count_nonzero(baseline_spike_bits_slice)
    freqinc_nr_spikes = np.count_nonzero(freqinc_spike_bits_slice)
    
    # NOT IN HZ!
    all_firing_rate.append(all_nr_spikes)
    baseline_firing_rate.append(baseline_nr_spikes) 
    freqinc_firing_rate.append(freqinc_nr_spikes) 

    
#since time goes from 0-1000 (1000 included, we need to take into account last case

i += 1
all_spike_bits_slice = spike_bits[:,i*jumps:spike_bits.shape[1]-1]
baseline_spike_bits_slice = spike_bits[0:30-nr_neurons,spike_bits.shape[1]-1]
freqinc_spike_bits_slice = spike_bits[30-nr_neurons:30,spike_bits.shape[1]-1]
    
    
all_nr_spikes = np.count_nonzero(all_spike_bits_slice)
baseline_nr_spikes = np.count_nonzero(baseline_spike_bits_slice)
freqinc_nr_spikes = np.count_nonzero(freqinc_spike_bits_slice)
    
# NOT IN HZ!
all_firing_rate.append(all_nr_spikes)
baseline_firing_rate.append(baseline_nr_spikes) 
freqinc_firing_rate.append(freqinc_nr_spikes)
 
    


#transform to numpy so that we can convert to Hz and normalize
    
all_firing_rate_np =  np.asarray(all_firing_rate, dtype=np.float32)   
baseline_firing_rate_np =  np.asarray(baseline_firing_rate, dtype=np.float32)    
freqinc_firing_rate_np =  np.asarray(freqinc_firing_rate, dtype=np.float32)    
 

#conversion to hz and normalize
all_firing_rate = (all_firing_rate_np * (1000/firing_rate_bin)) / 30
baseline_firing_rate = (baseline_firing_rate_np * (1000/firing_rate_bin)) / 30-nr_neurons
freqinc_firing_rate = (freqinc_firing_rate_np * (1000/firing_rate_bin)) / nr_neurons


kernel_std = 7
kernel_width = 50

    
gauss_filter = signal.gaussian(kernel_width, std=kernel_std)
gauss_filter = gauss_filter / np.sum(gauss_filter)

all_convolved_firing_rate = np.convolve( all_firing_rate , gauss_filter, mode='same'  )
baseline_convolved_firing_rate = np.convolve( baseline_firing_rate , gauss_filter, mode='same'  )
freqinc_convolved_firing_rate = np.convolve( freqinc_firing_rate , gauss_filter, mode='same'  )



fig, axs = plt.subplots(3, sharex=True, sharey=True, num=8, figsize = (19,10))
fig.suptitle('Firing Rates for ' + exp_name + " with bins of: " + str(firing_rate_bin) + " ms" +  "\n Gaussian Kernel with std = " + str(kernel_std) + " and kernel width = " + str(kernel_width) , fontsize=18)
axs[0].plot(np.arange(0,1000,firing_rate_bin) , all_convolved_firing_rate)
axs[1].plot(np.arange(0,1000,firing_rate_bin) , baseline_convolved_firing_rate)
axs[2].plot(np.arange(0,1000,firing_rate_bin) , freqinc_convolved_firing_rate)

axs[0].set_title("Firing rate of all neurons")
axs[1].set_title("Firing rate of baseline neurons")
axs[2].set_title("Firing rate of freqinc neurons")

fig.text(0.08, 0.5, 'Firing rate (Hz)', ha='center', va='center', rotation='vertical', fontsize=18 )
plt.xlabel('Time (ms)', fontsize=18)
plt.savefig(RESULTS_FOLDER + "\\" + "Populations firing rate",bbox_inches='tight')






############################################################################################################
############################################################################################################
# CORRELATION MATRIX


corr_bin = 1
jumps = int(100000 / len(np.arange(0,1000,corr_bin)))  #used later  

spike_bits_binned = np.zeros( (30, len(np.arange(0,1000,corr_bin))))

for i in range(len(np.arange(0,1000,corr_bin))-1):
    all_spike_bits_slice = spike_bits[:,i*jumps:(i+1)*jumps]
    sliced_spike_bits = np.sum(all_spike_bits_slice, axis=1)
    
    spike_bits_binned[: ,i ] = sliced_spike_bits
    
    
    
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


plt.figure(9, figsize=(10,8))
plt.matshow(upper_corr_matrix, fignum=9)
plt.plot(  [0,30] , [30-nr_neurons-0.5,30-nr_neurons-0.5], 'k'   )
plt.plot(  [30-nr_neurons-0.5,30-nr_neurons-0.5] , [-0.5,30] , 'k'  )

plt.colorbar()
plt.title("Correlation matrix of spiketrains for " + exp_name + " with bins of " + str(corr_bin) + " ms" + "\n" + "Mean of all spiketrains: " + str(full_mean))

plt.savefig(RESULTS_FOLDER + "\\" + "Correlation_Matrix",bbox_inches='tight')








"""



if (nr_neurons != 1):
    
    
    
    
    
    #CORRELATION MATRIX BETWEEN THE TWO DIFFERENT GROUPS
 
    masked_zeroed_corr_matrix = masked_zeroed_corr_matrix[0:30-nr_neurons, 30-nr_neurons:30]
    
    
    
    both_groups_mean= np.sum(masked_zeroed_corr_matrix)/ np.count_nonzero(masked_zeroed_corr_matrix)
    
    
    #plot and save
    plt.figure(10, figsize=(10,8))
    plt.matshow(upper_corr_matrix[0:30-nr_neurons, 30-nr_neurons:30], fignum=10)
    plt.colorbar()
    plt.title("Correlation matrix between groups for " + exp_name + " with bins of " + str(corr_bin) + " ms" + "\n" + "Mean between both groups: " + str(both_groups_mean))
    plt.xlabel('Freq inc group')
    plt.ylabel('Baseline group')
    plt.savefig(RESULTS_FOLDER + "\\" + "Correlation_Matrix_betweengroups",bbox_inches='tight')
    
    
    



    
    
    
    # CORRELATION MATRIXES FOR OTHER GROUPS
    
    #BASELINE
    
    
    corr_matrix = np.corrcoef(spike_bits_binned[0:30-nr_neurons])
    upper_mask =  np.tri(corr_matrix.shape[0], k=-1)
    upper_mask[np.diag_indices(upper_mask.shape[0])] = 1
    upper_corr_matrix = np.ma.array(corr_matrix, mask=upper_mask)
    
    #following calculations made for means
    zero_mask = np.ma.array(corr_matrix, mask=upper_mask, fill_value=0 )
    masked_zeroed_corr_matrix = zero_mask.filled()
    #calculate means
    baseline_group_mean= np.sum(masked_zeroed_corr_matrix)/ np.count_nonzero(masked_zeroed_corr_matrix)
    
    
    #plot and save
    plt.figure(11, figsize=(10,8))
    plt.matshow(upper_corr_matrix, fignum=11)
    plt.colorbar()
    plt.title("Correlation matrix of baseline frequency spiketrains for " + exp_name + " with bins of " + str(corr_bin) + " ms" + "\n" + "Mean of baseline freq: " + str(baseline_group_mean))
    
    plt.savefig(RESULTS_FOLDER + "\\" + "Correlation_Matrix_baseline",bbox_inches='tight')
    
    
    
    
    
    
    
    #FREQ INCREASE
    
    corr_matrix = np.corrcoef(spike_bits_binned[30-nr_neurons:30])
    upper_mask =  np.tri(corr_matrix.shape[0], k=-1)
    upper_mask[np.diag_indices(upper_mask.shape[0])] = 1
    upper_corr_matrix = np.ma.array(corr_matrix, mask=upper_mask)
    
    #following calculations made for means
    zero_mask = np.ma.array(corr_matrix, mask=upper_mask, fill_value=0 )
    masked_zeroed_corr_matrix = zero_mask.filled()
    #calculate means
    freqinc_group_mean= np.sum(masked_zeroed_corr_matrix)/ np.count_nonzero(masked_zeroed_corr_matrix)
    
    
    #plot and save
    plt.figure(12, figsize=(10,8))
    plt.matshow(upper_corr_matrix, fignum=12)
    plt.colorbar()
    plt.title("Correlation matrix of frequency increase spiketrains for " + exp_name + " with bins of " + str(corr_bin) + " ms" + "\n" + "Mean of freqinc: " + str(freqinc_group_mean))
    
    plt.savefig(RESULTS_FOLDER + "\\" + "Correlation_Matrix_freqinc",bbox_inches='tight')
    
    
    
    

"""



############################################################################################################
############################################################################################################
# VTH 








vth_checker = int(input("Enter 1 if you wish to plot membrane potential from thalamocortical (TC) neuron : "))



if (vth_checker):
    vth = spiketrainsv2[10]
    T = np.arange(0,1500.01,0.01)
    fig = plt.figure(13, figsize=(17,8))
    #we omit first two millisecond 
    plt.plot(T[200:len(T)], vth[200:len(vth)])
    fig.canvas.draw()
    yleft, yright = plt.ylim()
    plt.axvline( x=1000, color='r', linestyle='--')
    plt.title("Thalamocortical Membrane potential  " + exp_name)
    plt.xlabel('Time from movement (ms) ')
    plt.ylabel('Voltage (mV)')
    plt.savefig(RESULTS_FOLDER + "\\" + "TC membrane potential",bbox_inches='tight')
    
    
    
    
    
    
    t_before = int(input("Indicate in ms beginning of sim time  : "))
    t_after = int(input("Indicate in ms end of sim time  : "))
    
    fig = plt.figure(14, figsize=(19,10))
    plt.subplot(211)
    plt.plot(T[t_before*100:t_after*100], vth[t_before*100:t_after*100])
    fig.canvas.draw()
    plt.axvline( x=1000, color='r', linestyle='--')
    xleft, xright = plt.xlim()
    yleft, yright = plt.ylim()
    plt.title("Thalamocortical Membrane Potential for correlation: " + str(int(corr)/100))
    plt.ylabel('Voltage (mV)')
    ax = plt.gca()
    labels = np.array([int(item._x) for item in ax.get_xticklabels()])
    ax.set_xticklabels(labels-1000)
    #repeat process again
    
    plt.subplot(212)
    
    spike_count = 0
    all_spikes_grouped = []
    tot_amount_spikes = 0
    plt.title("Thalamocortical Membrane Potential")
    
    for i in range(all_spiketrains.size):
        data = all_spiketrains[i][0] #had to do this because of data bugs, contains group of spiketrains
        neurons, spikes = data.shape
        all_spikes_grouped.append(data.flatten())
        tot_amount_spikes += (data.flatten()).size
        for neu in range(neurons):
            #draw the vertical lines
            plt.plot([data[neu],data[neu]],[spike_count,spike_count+1], 'b')
            spike_count += 1
        
    
    
    
    
    all_spikes = np.sort(all_spikes)
    minpause = float(input("select min pause in ms to show in rasterplot: "))
    total_pause_count = 0 #coutns how many ms of total pauses
    ax = plt.gca()
    
    for i in range (len(all_spikes)-1):
     
        if (all_spikes[i+1] - all_spikes[i] >= minpause):
        
            """
            #plot the vertical lines between the pause points
            plt.plot( [all_spikes[i], all_spikes[i]], [0, spike_count  ], 'r--')
            plt.plot( [all_spikes[i+1], all_spikes[i+1]], [0, spike_count  ], 'r--')
            """
            
            #plot horizontal lines at the top 
            #plt.plot( [all_spikes[i], all_spikes[i+1]], [0, 0  ], 'r', linewidth=7.0)
            rect_length = (all_spikes[i+1] - all_spikes[i]) - 2.5
            rect_pause = plt.Rectangle((all_spikes[i]+2.5, spike_count+0.5),rect_length, 0.5, color='r')
            ax.add_patch(rect_pause)
            #plt.plot( [all_spikes[i]+2, all_spikes[i+1]-2], [spike_count+0.5, spike_count+0.5  ], 'r')
            
            #add pause to total
            total_pause_count += all_spikes[i+1] - all_spikes[i]
            
        #adding last pause (if there is such)
#simulation ends at 1000ms so we check if last spike has a significant difference with respect to that
    if (  (1000 - all_spikes[len(all_spikes)-1]) >= minpause  ):
        rect_length = (1000 - all_spikes[len(all_spikes)-1]) - 2.5
        rect_pause = plt.Rectangle((all_spikes[len(all_spikes)-1]+2.5, spike_count+0.5),rect_length, 0.5, color='r')
        ax.add_patch(rect_pause)    

    
    plt.axvline( x=1000, color='r', linestyle='--') 
    ax.set_xlim([xleft,xright])
    ax.set_xticklabels(labels-1000)
    plt.title("SNr input to TC")
    plt.xlabel('Time from movement (ms) ')
    plt.ylabel('Spike train #')
    plt.savefig(RESULTS_FOLDER + "\\" + "Spiketrains_with_membrane_potential",bbox_inches='tight')

    




############################################################################################################
############################################################################################################
# ISI AND ISI CV 


isi_stdev = np.zeros( (30) )
isi_mean = np.zeros( (30) )
spike_count = 0

for i in range(all_spiketrains.size):
    data = all_spiketrains[i][0] #had to do this because of data bugs, contains group of spiketrains
    neurons, spikes = data.shape
    if (neurons > 30): #done because with diffmoth spikes with 1 spiketrain have this bug
            neurons, spikes = spikes, neurons
        
    for neu in range(neurons):
        ordered_spikes = np.sort( data[neu]  )
        isi = np.diff(ordered_spikes)
        isi_stdev[spike_count] = np.std(isi)
        isi_mean[spike_count] = np.mean(isi)
        spike_count += 1



CV = isi_stdev / isi_mean
CV_baseline = CV[0:30-nr_neurons]
CV_freqinc = CV[30-nr_neurons:30]

CV_baseline_mean = np.mean(CV_baseline)
CV_freqinc_mean = np.mean(CV_freqinc)
CV_both_groups = np.mean(CV)



CV_to_plot = [CV_baseline_mean, CV_freqinc_mean, CV_both_groups]
CV_labels = ["Baseline", "Freq Increase", "All neurons"]
CV_err = [np.std(CV_baseline), np.std(CV_freqinc), np.std(CV)]


plt.figure(16, figsize=(10,9))
plt.title("ISI CV Means")
plt.bar(range(3), CV_to_plot, tick_label=CV_labels, width=0.6, edgecolor='k', yerr=CV_err)

ax = plt.gca()


yleft, yright = plt.ylim()
ax.set_ylim([yleft,yright*1.3])
plt.savefig(RESULTS_FOLDER + "\\" + "ISI_CV",bbox_inches='tight')


plt.show()












