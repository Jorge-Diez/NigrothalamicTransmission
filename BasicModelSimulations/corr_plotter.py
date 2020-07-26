# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.io as sio
import numpy as np
import os
import pickle
import warnings
from natsort import natsorted
from mpl_toolkits import mplot3d



root_folder = input("please enter name of the root folder: ")
nr_experiments = int(input("Total number of frequency experiments: "))

warnings.filterwarnings("ignore") #due to matplotlib depreciation warnings


with open('corrmatrixes' +  root_folder + '.pickle', 'rb') as handle:
    ALL_CORR_MAT_LIST = (pickle.load(handle))
        
with open('corrdiffmatrixes' +  root_folder + '.pickle', 'rb') as handle:
    ALL_CORRDIFF_MAT_LIST = (pickle.load(handle))
    
with open('corrpercmatrixes' +  root_folder + '.pickle', 'rb') as handle:
   ALL_CORRPERC_MAT_LIST = (pickle.load(handle))
   
RESULTS_FOLDER = "CORR_GRAPHIC_RESULTS"
RESULTS_FOLDER_CORR_VALUES = "CORR_GRAPHIC_RESULTS\\CORR_VALUES"
RESULTS_FOLDER_CORR_DIFFERENCES = "CORR_GRAPHIC_RESULTS\\CORR_DIFFERENCES"
RESULTS_FOLDER_CORR_PERC = "CORR_GRAPHIC_RESULTS\\CORR_PERCENTAGES"

all_folder_results = [RESULTS_FOLDER, RESULTS_FOLDER_CORR_VALUES, RESULTS_FOLDER_CORR_DIFFERENCES, RESULTS_FOLDER_CORR_PERC]

for i in all_folder_results:
    if not os.path.exists(i):
        os.mkdir(i)


   
corrs = [30, 40, 50, 60, 70, 80, 90, 100]


root_folder_check = int(input("Enter 1 if you want all graphs first: "))

if (root_folder_check):
    for i,j in zip(corrs,ALL_CORR_MAT_LIST):
        x_values = np.arange(51,51+nr_experiments).astype(str)
    
    
    
        fig = plt.figure(figsize=(19,10))
        plt.imshow(j, cmap='jet', vmin = 0, vmax = 1, aspect='auto')
        cbar = plt.colorbar()
        
        ax = plt.axes()
        plt.title("Actual Correlation values for correlation of " + str(i/100), fontsize = 20)
        plt.xlabel("Frequency increase to ", fontsize = 16)
        plt.ylabel("Nr_neurons_freq_inc", fontsize = 16)
        cbar.set_label('Correlation', fontsize = 20)
        
        plt.yticks( np.arange(0,30,1), np.arange(1,31,1))    
        plt.xticks( np.arange(0,nr_experiments,1), x_values) 
        
        # Minor ticks
        ax.set_yticks(np.arange(0.5,30,1), minor=True);
        ax.set_xticks(np.arange(0.5,nr_experiments,1), minor=True);
        ax.tick_params(axis=u'both', which=u'both',length=0)
        
        ax.grid(color='k', which='minor')
        
        plt.savefig(RESULTS_FOLDER_CORR_VALUES + "\\" + "CORR_VALUE_" + str(i) ,bbox_inches='tight')
        plt.close()
        
        
        ### NOW IN 3D
        
        
        fig = plt.figure(figsize=(19,10))
    
        X = np.arange(0, nr_experiments)
        Y = np.arange(0, 30)
        
        xx,yy = np.meshgrid(X,Y)
        ax = plt.axes(projection='3d')
        ax.view_init(elev=15., azim=220)
        
        
        surf = ax.plot_surface(xx, yy, j, cmap='jet', linewidth=0, antialiased=False, vmin = 0, vmax = 1)
        plt.title("Actual Correlation values for correlation of " + str(i/100), fontsize = 20)
        plt.xlabel("Frequency increase to ", fontsize = 16)
        plt.ylabel("Nr_neurons_freq_inc", fontsize = 16)
        ax.set_zlabel("Correlation", fontsize = 16)
        
        fig.colorbar(surf, shrink = 1)
        
        plt.yticks( np.arange(0,30,3), np.arange(1,31,3))    
        plt.xticks( np.arange(0,nr_experiments,4), np.arange(51,51+nr_experiments,4))   
        
        plt.savefig(RESULTS_FOLDER_CORR_VALUES + "\\" + "CORR_VALUE_3D_" + str(i) ,bbox_inches='tight')
        plt.close()
        
        
        
        
        
    
    for i,j in zip(corrs,ALL_CORRDIFF_MAT_LIST):
        x_values = np.arange(51,51+nr_experiments).astype(str)
    
    
    
        fig = plt.figure(figsize=(19,10))
        plt.imshow(j, cmap='jet', aspect='auto')
        cbar = plt.colorbar()
        
        ax = plt.axes()
        plt.title("Correlation difference values for correlation of " + str(i/100), fontsize = 20)
        plt.xlabel("Frequency increase to ", fontsize = 16)
        plt.ylabel("Nr_neurons_freq_inc", fontsize = 16)
        cbar.set_label('Correlation Difference', fontsize = 20)
        
        plt.yticks( np.arange(0,30,1), np.arange(1,31,1))    
        plt.xticks( np.arange(0,nr_experiments,1), x_values) 
        
        # Minor ticks
        ax.set_yticks(np.arange(0.5,30,1), minor=True);
        ax.set_xticks(np.arange(0.5,nr_experiments,1), minor=True);
        ax.tick_params(axis=u'both', which=u'both',length=0)
        
        ax.grid(color='k', which='minor')
        
        plt.savefig(RESULTS_FOLDER_CORR_DIFFERENCES + "\\" + "CORR_DIFF_" + str(i) ,bbox_inches='tight')
        plt.close()
        
        
        ### NOW IN 3D
        
        
        fig = plt.figure(figsize=(19,10))
    
        X = np.arange(0, nr_experiments)
        Y = np.arange(0, 30)
        
        xx,yy = np.meshgrid(X,Y)
        ax = plt.axes(projection='3d')
        ax.view_init(elev=15., azim=220)
        
        
        surf = ax.plot_surface(xx, yy, j, cmap='jet', linewidth=0, antialiased=False)
        plt.title("Correlation difference values for correlation of " + str(i/100), fontsize = 20)
        plt.xlabel("Frequency increase to ", fontsize = 16)
        plt.ylabel("Nr_neurons_freq_inc", fontsize = 16)
        ax.set_zlabel("Correlation Difference", fontsize = 16)
        
        fig.colorbar(surf, shrink = 1)
        
        plt.yticks( np.arange(0,30,3), np.arange(1,31,3))    
        plt.xticks( np.arange(0,nr_experiments,4), np.arange(51,51+nr_experiments,4))   
        
        plt.savefig(RESULTS_FOLDER_CORR_DIFFERENCES + "\\" + "CORR_DIFF_3D_" + str(i) ,bbox_inches='tight')
        plt.close()
        
        
        
        
        
        
    for i,j in zip(corrs,ALL_CORRPERC_MAT_LIST):
        x_values = np.arange(51,51+nr_experiments).astype(str)
    
    
    
        fig = plt.figure(figsize=(19,10))
        plt.imshow(j, cmap='jet', aspect='auto')
        cbar = plt.colorbar()
        
        ax = plt.axes()
        plt.title("Correlation percentage difference values for correlation of " + str(i/100), fontsize = 20)
        plt.xlabel("Frequency increase to ", fontsize = 16)
        plt.ylabel("Nr_neurons_freq_inc", fontsize = 16)
        cbar.set_label('Correlation Difference in %', fontsize = 20)
        
        plt.yticks( np.arange(0,30,1), np.arange(1,31,1))    
        plt.xticks( np.arange(0,nr_experiments,1), x_values) 
        
        # Minor ticks
        ax.set_yticks(np.arange(0.5,30,1), minor=True);
        ax.set_xticks(np.arange(0.5,nr_experiments,1), minor=True);
        ax.tick_params(axis=u'both', which=u'both',length=0)
        
        ax.grid(color='k', which='minor')
        
        plt.savefig(RESULTS_FOLDER_CORR_PERC + "\\" + "CORR_PERC_" + str(i) ,bbox_inches='tight')
        plt.close()
        
        
        ### NOW IN 3D
        
        
        fig = plt.figure(figsize=(19,10))
    
        X = np.arange(0, nr_experiments)
        Y = np.arange(0, 30)
        
        xx,yy = np.meshgrid(X,Y)
        ax = plt.axes(projection='3d')
        ax.view_init(elev=15., azim=220)
        
        
        surf = ax.plot_surface(xx, yy, j, cmap='jet', linewidth=0, antialiased=False)
        plt.title("Correlation percentage difference values for correlation of " + str(i/100), fontsize = 20)
        plt.xlabel("Frequency increase to ", fontsize = 16)
        plt.ylabel("Nr_neurons_freq_inc", fontsize = 16)
        ax.set_zlabel("Correlation Difference in %", fontsize = 16)
        
        fig.colorbar(surf, shrink = 1)
        
        plt.yticks( np.arange(0,30,3), np.arange(1,31,3))    
        plt.xticks( np.arange(0,nr_experiments,4), np.arange(51,51+nr_experiments,4))   
        
        plt.savefig(RESULTS_FOLDER_CORR_PERC + "\\" + "CORR_PERC_3D_" + str(i) ,bbox_inches='tight')
        plt.close()





freq_plot = int(input("Enter frequency : "))
method = input("Enter method of simulation: ")
freq_index = freq_plot-51 #so we can plot easily

labels = ["Corr. Incr. to 0.3", "Corr. Incr. to 0.4", "Corr. Incr. to 0.5", "Corr. Incr. to 0.6",
          "Corr. Incr. to 0.7", "Corr. Incr. to 0.8", "Corr. Incr. to 0.9", "Corr. Incr. to 1",]

plt.figure(1, figsize=(10,9))
for i in range(len(ALL_CORR_MAT_LIST)):
    
    all_data = ALL_CORR_MAT_LIST[i]
    corr_data = all_data[:,freq_index]
    
    plt.plot(corr_data)




plt.title("Actual Correlation values for Frequency:  " + str(freq_plot) + " with " + method  + " method", fontsize = 20)
plt.xlabel("Nr_neurons_freq_inc ", fontsize = 16)
plt.ylabel("Actual correlation value", fontsize = 16)
plt.legend(labels)
plt.savefig(RESULTS_FOLDER + "\\" + "CORR_FOR_FREQ" + str(freq_plot) +  "_" + method ,bbox_inches='tight')





plt.figure(2, figsize=(10,9))
for i in range(len(ALL_CORRPERC_MAT_LIST)):
    
    all_data = ALL_CORRPERC_MAT_LIST[i]
    corr_data = all_data[:,freq_index]
    
    plt.plot(corr_data)




plt.title(" Correlation Difference in % for Frequency:  " + str(freq_plot) + " with " + method  + " method", fontsize = 20)
plt.xlabel("Nr_neurons_freq_inc ", fontsize = 16)
plt.ylabel("Correlation Difference in %", fontsize = 16)
plt.legend(labels)
plt.savefig(RESULTS_FOLDER + "\\" + "CORRPERC_FOR_FREQ" + str(freq_plot) +  "_" + method ,bbox_inches='tight')
















corr_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8 , 0.9, 1]
corr = float(input("Enter intended correlation : "))
nr_neurons = int(input("Enter number of neurons : "))
neuron_index = nr_neurons-1 #so we can plot easily
corr_index = corr_values.index(corr)

plt.figure(3, figsize=(10,9))



all_nr_neurons = [5, 10, 15, 20, 25]
labels = ["5 neurons", "10 neurons", "15 neurons", "20 neurons", "25 neurons"] 

for i,nr_neurons in enumerate(all_nr_neurons):
    neuron_index = nr_neurons-1 #so we can plot easily
    all_data = ALL_CORR_MAT_LIST[corr_index]
    corr_data = all_data[neuron_index,:]
    plt.plot(corr_data)
    
    
plt.title("Actual Correlation values for Nr_neurons:  " + str(nr_neurons) + " with " + method  + " method", fontsize = 20)
plt.xlabel("Freq Increase to ", fontsize = 16)
plt.ylabel("Actual correlation value", fontsize = 16)
plt.xticks( np.arange(0,nr_experiments,4), np.arange(51,51+nr_experiments,4))   
xleft, xright = plt.xlim()
plt.plot( [xleft, xright], [corr, corr  ], 'r--')
plt.legend(labels)
plt.savefig(RESULTS_FOLDER + "\\" + "CORR_FOR_INTENDED_CORR" + str(int(corr*100)) +  "_" + method ,bbox_inches='tight')


plt.figure(4, figsize=(10,9))

for i,nr_neurons in enumerate(all_nr_neurons):
    neuron_index = nr_neurons-1 #so we can plot easily
    all_data = ALL_CORRPERC_MAT_LIST[corr_index]
    corr_data = all_data[neuron_index,:]
    plt.plot(corr_data)
    
    
plt.title("Correlation Difference in % for Nr_neurons:  " + str(nr_neurons) + " with " + method  + " method", fontsize = 20)
plt.xlabel("Freq Increase to ", fontsize = 16)
plt.ylabel("Correlation Difference in %", fontsize = 16)
plt.xticks( np.arange(0,nr_experiments,4), np.arange(51,51+nr_experiments,4))   
plt.legend(labels)
plt.savefig(RESULTS_FOLDER + "\\" + "CORRPERC_FOR_INTENDED_CORR" + str(int(corr*100)) +  "_" + method ,bbox_inches='tight')






