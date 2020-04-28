#!/usr/bin/python

import os
from natsort import natsorted
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as sio

pwd = os.getcwd()


#folder where our results will be located
RESULTS_FOLDER = "graphic_results"


if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)
    

#getting the corresponding directory for the basic experiment with 50 hz firing rate for all neurons
basic_exp_folder = "baseline 50hz 30 neurons"
main_basic_exp_folder = os.path.join(pwd, basic_exp_folder)


#obtain the corresponding correlation values and the X acis plotting values
mat_try = sio.loadmat(main_basic_exp_folder + '\\plotting-params-MIP')
corrs = mat_try["corr_val"]
OVR = mat_try ["OVR"]

fig = plt.figure(figsize=(10,8))
                 
plt.imshow(OVR, cmap='jet', extent=[0.3,1,0,1], vmin = 0, vmax = 1)
cbar = plt.colorbar()

#set the image to display correctly + title and axes
ax = plt.axes()
ax.axes.get_yaxis().set_visible(False)
plt.title("Baseline 30 Neurons 50 Hz", fontsize = 20)
plt.xlabel("Correlation Coefficient", fontsize = 16)
cbar.set_label('Tranmission Quality', fontsize = 20)

#save figure without the corresponding "white space" around it
plt.savefig(RESULTS_FOLDER + '\\baseline50hz',bbox_inches='tight')
plt.close()



##Now for generating the actual results


big_exp_folder = "BIG_EXPERIMENTS_BASE_50HZ"
main_big_exp_folder = os.path.join(pwd, big_exp_folder)
x_values = ["0.3","0.4","0.5","0.6","0.7","0.8","0.9","1"]


for i, experiment_name in enumerate(os.listdir(main_big_exp_folder)):
    #each of these is a folder containing a experiment
    #where the number of neurons with an increased frequency increases by 1 by folder
    OVR_RESULTS = np.zeros((30,8)) #all results will be pooled here
    
    exp_folder = os.path.join(main_big_exp_folder, experiment_name)
    fig = plt.figure(figsize=(19,10))
    print(exp_folder)
    
    for j,nr_neurons_increased in enumerate(natsorted(os.listdir(os.path.join(main_big_exp_folder, experiment_name)))):
        #each of these is a folder where number of neurons was increased
        temp_neuron_nr_folder = os.path.join(exp_folder, nr_neurons_increased)  
        mat_data = sio.loadmat(temp_neuron_nr_folder + '\\plotting-params-MIP')#load data from nr_neuron sub experiment
        
        OVR_RESULTS[j:] = mat_data["OVR"]
        
    plt.imshow(OVR_RESULTS, cmap='jet', vmin = 0, vmax = 1, aspect='auto')
    cbar = plt.colorbar()
    
    ax = plt.axes()
    plt.title(experiment_name, fontsize = 20)
    plt.xlabel("Correlation Coefficient", fontsize = 16)
    plt.ylabel("Nr_neurons_freq_inc", fontsize = 16)
    cbar.set_label('Tranmission Quality', fontsize = 20)
    
    plt.yticks( np.arange(0,30,1), np.arange(1,31,1))    
    plt.xticks( np.arange(0,8,1), x_values) 
    
    # Minor ticks
    ax.set_yticks(np.arange(0.5,30,1), minor=True);
    ax.set_xticks(np.arange(0.5,8,1), minor=True);
    ax.tick_params(axis=u'both', which=u'both',length=0)
    
    ax.grid(color='k', which='minor')
    
    plt.savefig(RESULTS_FOLDER + '\\' + experiment_name,bbox_inches='tight')
    plt.close()

