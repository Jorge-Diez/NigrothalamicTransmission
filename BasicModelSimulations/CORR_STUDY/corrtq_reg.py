import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter 


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



fig = plt.figure(1, figsize=(19,10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=15., azim=300)
ax.scatter(freq_indeces, corr_results, tq_results,  s=10)
plt.title("tq and correlation scatterplot for: " + title)
ax.set_xlabel("Firing rate incICErease ", fontsize = 16)
ax.set_ylabel("Corr", fontsize = 16)
ax.set_zlabel("TQ", fontsize = 16)

yleft, yright = plt.ylim()
ax = plt.gca()
ax.set_ylim([-0.05,1.05])



combos = list(zip(freq_indeces, tq_results))
weight_counter = Counter(combos)
weights = [weight_counter[(freq_indeces[i], tq_results[i])] for i, _ in enumerate(freq_indeces)]
weights = np.array(weights)

plt.figure(2, figsize=(11,7))
plt.scatter(freq_indeces, tq_results,  s=weights*1.2)
plt.title("Frequency and TQ scatterplot for " + title, fontsize = 18)
plt.xlabel("Frequency Increase (Hz)", fontsize = 16)
plt.xticks(fontsize= 14)
plt.ylabel("Transmission quality", fontsize = 16)
plt.yticks(fontsize= 14)
yleft, yright = plt.ylim()
ax = plt.gca()
ax.set_ylim([-0.05,1.05])
plt.savefig("freq_tq_scatter_" + title ,bbox_inches='tight')




combos = list(zip(corr_results, tq_results))
weight_counter = Counter(combos)
weights = [weight_counter[(corr_results[i], tq_results[i])] for i, _ in enumerate(corr_results)]
weights = np.array(weights)


plt.figure(3, figsize=(11,7))
plt.scatter(corr_results, tq_results,  s=weights*1.2)
plt.title("Correlation and TQ scatterplot for " + title, fontsize = 18)
plt.xlabel("Correlation after DBS", fontsize = 16)
plt.xticks(fontsize= 14)
plt.ylabel("Transmission quality", fontsize = 16)
plt.yticks(fontsize= 14)
yleft, yright = plt.ylim()
ax = plt.gca()
ax.set_ylim([-0.05,1.05])
plt.savefig("corr_tq_scatter_" + title ,bbox_inches='tight')
