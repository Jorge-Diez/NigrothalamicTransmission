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



neural_data = sio.loadmat(main_basic_exp_folder + '\\voltage-traces-and-inputs\\30-70')["mem_v_traces"]

spiketrains = neural_data[0] # grab the main structure mem_v_traces
spiketrainsv2 = spiketrains[0]
all_spiketrains = spiketrainsv2[1].flatten()



spike_count = 0



for i in range(all_spiketrains.size):
    data = all_spiketrains[i][0] #had to do this because of data bugs
    neurons, spikes = data.shape
    for neu in range(neurons):
        plt.plot([data[neu],data[neu]],[spike_count,spike_count+1], 'b')
        spike_count += 1



#add graph information
plt.title('Spike raster plot')
plt.xlabel('Spikes (ms)')
plt.ylabel('Neurons')

 

plt.show()
