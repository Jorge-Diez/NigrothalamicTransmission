# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import os
import warnings


pwd = os.getcwd()
warnings.filterwarnings("ignore") #due to matplotlib depreciation warnings



folder_results = input("please enter the directory where the results are located:")


main_basic_exp_folder = os.path.join(pwd, folder_results) #get main experiment folder

corr = input("please indicate correlation: ")

neural_data = sio.loadmat(main_basic_exp_folder + '\\voltage-traces-and-inputs\\' + corr + '-70')["mem_v_traces"]

spiketrains = neural_data[0] # grab the main structure mem_v_traces
spiketrainsv2 = spiketrains[0]
all_spiketrains = spiketrainsv2[1].flatten()



spike_count = 0
all_spikes = 0

plt.figure(1)
for i in range(all_spiketrains.size):
    data = all_spiketrains[i][0] #had to do this because of data bugs
    neurons, spikes = data.shape
    all_spikes = data.flatten()
    for neu in range(neurons):
        plt.plot([data[neu],data[neu]],[spike_count,spike_count+1], 'b')
        spike_count += 1


#add graph information
plt.title('Spike raster plot')
plt.xlabel('Time (ms)')
plt.ylabel('Neurons')


#amplitude distribution
# 0.01 ms bin 
t_vec = np.arange(0,1000.01,0.01)
hist, bins = np.histogram(data, t_vec)




plt.figure(2)
plt.hist(hist[hist != 0], range(31), ec='black')
locs, labels = plt.yticks()
plt.title("Amplitude distribution")




bars, bars_counts = np.unique(np.sort(hist), return_counts = True)
#delete cases for 0
bars_full = np.zeros(31)

bars = np.delete(bars, 0)
bars_counts = np.delete(bars_counts, 0)
bars_full[bars] = bars_counts

probs_full = bars_full/bars_full.sum()

plt.figure(3)
plt.title("Amplitude-Frequency distribution")
plt.bar(np.arange(len(bars_full)),probs_full)
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
plt.figure(4)
plt.title("MIP with bins")
plt.bar(np.arange(len(bars_bins)),probs_bins)
plt.xticks(np.arange(len(bars_bins)), bars_bins)






plt.show()
