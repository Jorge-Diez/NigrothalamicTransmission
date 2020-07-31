import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D



exp_name = input("Enter name of the experiment: ")

tq_results_name = "all_trials_TQ_results_" + exp_name + ".txt"
corr_results_name = "all_trials_corr_results_" + exp_name + ".txt"


with open(tq_results_name) as f:
    tq_results = f.readlines()
f.close()
  

tq_results = [ el.replace(',','.') for el in tq_results   ]
tq_results = np.asarray(tq_results, dtype=np.float32)
    
corr_results = np.loadtxt(corr_results_name)



nr_neuron_experiments = int(input("number of neuron experiments: "))
title = input("Experiment name: ")


corr_indeces = np.tile(([0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9, 1]) , int(np.size(corr_results)/8))
freq_indeces = np.repeat(np.arange(51,91) , int(np.size(corr_results)/np.size(np.arange(51,91))))
nr_neur_indeces = np.repeat ( np.arange(1,nr_neuron_experiments+1), 8   )
nr_neur_indeces = np.tile(nr_neur_indeces, 40)



#ax = fig.add_subplot(111, projection='3d')
plt.figure(figsize=(19,10))
plt.scatter(tq_results, corr_results,  s=10)
plt.title("tq and correlation scatterplot for: " + title)
plt.xlabel("TQ ", fontsize = 16)
plt.ylabel("Corr", fontsize = 16)

yleft, yright = plt.ylim()
ax = plt.gca()
ax.set_ylim([-0.05,1.05])