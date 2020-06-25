#!/usr/bin/python

import os
from natsort import natsorted
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as sio
import warnings
from mpl_toolkits import mplot3d


def progressbar(acc, total, total_bar):
    """ simple progressbar """
    frac = acc/total
    filled_progbar = round(frac*total_bar)
    print('\r', '#'*filled_progbar + '-'*(total_bar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')



pwd = os.getcwd()
warnings.filterwarnings("ignore") #due to matplotlib depreciation warnings


#folder where our results will be located
RESULTS_FOLDER = "graphic_results"
RESULTS_FOLDER_TQ = "graphic_results\\TQ_VALUES"
RESULTS_FOLDER_TQ_DIFF = "graphic_results\\TQ_DIFF"

all_folder_results = [RESULTS_FOLDER, RESULTS_FOLDER_TQ, RESULTS_FOLDER_TQ_DIFF ]

for i in all_folder_results:
    if not os.path.exists(i):
        os.mkdir(i)
        
    
    

#getting the corresponding directory for the basic experiment with 50 hz firing rate for all neurons
basic_exp_folder = "baseline 50hz 30 neurons"
main_basic_exp_folder = os.path.join(pwd, basic_exp_folder)


#obtain the corresponding correlation values and the X acis plotting values
mat_try = sio.loadmat(main_basic_exp_folder + '\\plotting-params-MIP')
corrs = mat_try["corr_val"]
OG_OVR = mat_try ["OVR"]

fig = plt.figure(figsize=(10,8))
                 
plt.imshow(OG_OVR, cmap='jet', extent=[0.3,1,0,1], vmin = 0, vmax = 1)
cbar = plt.colorbar()

print("TQ OF BASELINE IS: {}".format(OG_OVR))
print("MEAN TQ OF BASELINE IS: {}".format(np.mean(OG_OVR)))

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


big_exp_folder = input("Name of folder with data: ")
main_big_exp_folder = os.path.join(pwd, big_exp_folder)
x_values = ["0.3","0.4","0.5","0.6","0.7","0.8","0.9","1"]

nr_experiments = int(input("Total number of frequency experiments: "))


#results are saved like this so that we can later obtain maximums and minimums from all results
#it may be slightly less inefficient but i just want some results
ALL_TITLES = [] #used for plotting and saving titles
ALL_OVR_RESULTS = []
ALL_OVR_RESULTS_DIFFERENCE = []
ALL_MEAN_OVR = np.zeros((30,nr_experiments))
ALL_MEAN_OVR_DIFFERENCE = np.zeros((30,nr_experiments))

#Im doing this with a migraine so i want to make it simple
ALL_OVR_CORR_30 = np.zeros((30,nr_experiments))
ALL_OVR_CORR_40 = np.zeros((30,nr_experiments))
ALL_OVR_CORR_50 = np.zeros((30,nr_experiments))
ALL_OVR_CORR_60 = np.zeros((30,nr_experiments))
ALL_OVR_CORR_70 = np.zeros((30,nr_experiments))
ALL_OVR_CORR_80 = np.zeros((30,nr_experiments))
ALL_OVR_CORR_90 = np.zeros((30,nr_experiments))
ALL_OVR_CORR_100 = np.zeros((30,nr_experiments))

print("Obtaining results.... ")
for i, experiment_name in enumerate(os.listdir(main_big_exp_folder)):
    #each of these is a folder containing a experiment
    #where the number of neurons with an increased frequency increases by 1 by 
    
    OVR_RESULTS = np.zeros((30,8)) #all TQ results will be pooled here
    OVR_RESULTS_DIFFERENCE = np.zeros((30,8)) #TQ diferrence with respect to baseline

    
    exp_folder = os.path.join(main_big_exp_folder, experiment_name)
    
    for j,nr_neurons_increased in enumerate(natsorted(os.listdir(os.path.join(main_big_exp_folder, experiment_name)))):
        #each of these is a folder where number of neurons was increased
        temp_neuron_nr_folder = os.path.join(exp_folder, nr_neurons_increased)  
        mat_data = sio.loadmat(temp_neuron_nr_folder + '\\plotting-params-MIP')#load data from nr_neuron sub experiment
        
        #save results from experiment with specific neurons
        OVR_RESULTS[j:] = mat_data["OVR"]
        OVR_RESULTS_DIFFERENCE[j:] = mat_data["OVR"] - OG_OVR
        ALL_MEAN_OVR[j,i] = np.mean(mat_data["OVR"])
        ALL_MEAN_OVR_DIFFERENCE[j,i] = np.mean(mat_data["OVR"] - OG_OVR)
        # TODO do all mean for difference and changing order of mean 

    #Save results from main experiment
    ALL_OVR_RESULTS.append(OVR_RESULTS)    
    ALL_OVR_RESULTS_DIFFERENCE.append(OVR_RESULTS_DIFFERENCE)
    ALL_TITLES.append(experiment_name)
    

    
 

    

#time to plot the results
tq_diff_min = np.amin(OVR_RESULTS_DIFFERENCE)
tq_diff_max = np.amax(OVR_RESULTS_DIFFERENCE)
print("Results calculated, proceeding to plot and save.... ")


for i in range (nr_experiments):
    
    
    
    #I personally prefer not doing a function for plotting two times
    OVR_RESULTS = ALL_OVR_RESULTS[i]
    OVR_RESULTS_DIFFERENCE = ALL_OVR_RESULTS_DIFFERENCE[i]
    experiment_name = ALL_TITLES[i]
    
    ###############################################################################   
    # PLOT THE TQ VALUES FIRST
    ###############################################################################
    fig = plt.figure(figsize=(19,10))    
    plt.imshow(OVR_RESULTS, cmap='jet', vmin = 0, vmax = 1, aspect='auto')
    cbar = plt.colorbar()
    
    ax = plt.axes()
    plt.title(experiment_name, fontsize = 20)
    plt.xlabel("Correlation Coefficient", fontsize = 16)
    plt.ylabel("Nr_neurons_freq_inc", fontsize = 16)
    cbar.set_label('Tranmission Quality', fontsize = 20)
    
    #changing ticks
    plt.yticks( np.arange(0,30,1), np.arange(1,31,1))    
    plt.xticks( np.arange(0,8,1), x_values) 
    
    # Minor ticks for gridlines
    ax.set_yticks(np.arange(0.5,30,1), minor=True);
    ax.set_xticks(np.arange(0.5,8,1), minor=True);
    ax.tick_params(axis=u'both', which=u'both',length=0)
    
    ax.grid(color='k', which='minor')
    
    plt.savefig(RESULTS_FOLDER_TQ + "\\" +  experiment_name,bbox_inches='tight')
    plt.close()
    
    
    ###############################################################################   
    # PLOT THE TQ DIFF VALUES
    ###############################################################################
    
    
    fig = plt.figure(figsize=(19,10))
    plt.imshow(OVR_RESULTS_DIFFERENCE, cmap='jet', vmin = tq_diff_min, vmax = tq_diff_max, aspect='auto')
    cbar = plt.colorbar()
    
    ax = plt.axes()
    plt.title(experiment_name, fontsize = 20)
    plt.xlabel("Correlation Coefficient", fontsize = 16)
    plt.ylabel("Nr_neurons_freq_inc", fontsize = 16)
    cbar.set_label('Differences with respect to baseline', fontsize = 20)
    
    plt.yticks( np.arange(0,30,1), np.arange(1,31,1))    
    plt.xticks( np.arange(0,8,1), x_values) 
    
    # Minor ticks
    ax.set_yticks(np.arange(0.5,30,1), minor=True);
    ax.set_xticks(np.arange(0.5,8,1), minor=True);
    ax.tick_params(axis=u'both', which=u'both',length=0)
    
    ax.grid(color='k', which='minor')
    
    plt.savefig(RESULTS_FOLDER_TQ_DIFF + "\\" + experiment_name + "DIFF",bbox_inches='tight')
    plt.close()
    

    progressbar(i+1, nr_experiments, 50)











##Now we save the MEAN results
#in 2D
        

x_values = np.arange(51,51+nr_experiments).astype(str)



fig = plt.figure(figsize=(19,10))
plt.imshow(ALL_MEAN_OVR, cmap='jet', vmin = 0, vmax = 1, aspect='auto')
cbar = plt.colorbar()

ax = plt.axes()
plt.title("TQ mean over nr_neurons", fontsize = 20)
plt.xlabel("Frequency increase to ", fontsize = 16)
plt.ylabel("Nr_neurons_freq_inc", fontsize = 16)
cbar.set_label('TQ mean', fontsize = 20)

plt.yticks( np.arange(0,30,1), np.arange(1,31,1))    
plt.xticks( np.arange(0,nr_experiments,1), x_values) 

# Minor ticks
ax.set_yticks(np.arange(0.5,30,1), minor=True);
ax.set_xticks(np.arange(0.5,nr_experiments,1), minor=True);
ax.tick_params(axis=u'both', which=u'both',length=0)

ax.grid(color='k', which='minor')

plt.savefig(RESULTS_FOLDER + "\\" + "MEAN_2D",bbox_inches='tight')
plt.close()   




#in 3D
fig = plt.figure(figsize=(19,10))

X = np.arange(0, nr_experiments)
Y = np.arange(0, 30)

xx,yy = np.meshgrid(X,Y)
ax = plt.axes(projection='3d')
ax.view_init(elev=15., azim=220)


surf = ax.plot_surface(xx, yy, ALL_MEAN_OVR, cmap='jet', linewidth=0, antialiased=False, vmin = 0, vmax = 1)
plt.title("TQ mean over nr_neurons", fontsize = 20)
ax.set_xlabel("Frequency increase to ", fontsize = 16)
ax.set_ylabel("Nr_neurons_freq_inc", fontsize = 16)
ax.set_zlabel("TQ mean", fontsize = 16)

fig.colorbar(surf, shrink = 1)

plt.yticks( np.arange(0,30,3), np.arange(1,31,3))    
plt.xticks( np.arange(0,nr_experiments,4), np.arange(51,51+nr_experiments,4))   

plt.savefig(RESULTS_FOLDER + "\\" + "MEAN_3D",bbox_inches='tight')
plt.close()












##Now we save the MEAN DIFFERENCE results
#in 2D
        


x_values = np.arange(51,51+nr_experiments).astype(str)



fig = plt.figure(figsize=(19,10))
plt.imshow(ALL_MEAN_OVR_DIFFERENCE, cmap='jet', vmin = 0, vmax = 1, aspect='auto')
cbar = plt.colorbar()

ax = plt.axes()
plt.title("TQ difference mean over nr_neurons", fontsize = 20)
plt.xlabel("Frequency increase to ", fontsize = 16)
plt.ylabel("Nr_neurons_freq_inc", fontsize = 16)
cbar.set_label('Difference with respect to baseline', fontsize = 20)

plt.yticks( np.arange(0,30,1), np.arange(1,31,1))    
plt.xticks( np.arange(0,nr_experiments,1), x_values) 

# Minor ticks
ax.set_yticks(np.arange(0.5,30,1), minor=True);
ax.set_xticks(np.arange(0.5,nr_experiments,1), minor=True);
ax.tick_params(axis=u'both', which=u'both',length=0)

ax.grid(color='k', which='minor')

plt.savefig(RESULTS_FOLDER + "\\" + "MEAN_2D_DIFFERENCE",bbox_inches='tight')
plt.close()   




#in 3D
fig = plt.figure(figsize=(19,10))

X = np.arange(0, nr_experiments)
Y = np.arange(0, 30)

xx,yy = np.meshgrid(X,Y)
ax = plt.axes(projection='3d')
ax.view_init(elev=15., azim=220)


surf = ax.plot_surface(xx, yy, ALL_MEAN_OVR_DIFFERENCE, cmap='jet', linewidth=0, antialiased=False, vmin = 0, vmax = 1)
plt.title("TQ difference mean over nr_neurons", fontsize = 20)
ax.set_xlabel("Frequency increase to ", fontsize = 16)
ax.set_ylabel("Nr_neurons_freq_inc", fontsize = 16)
ax.set_zlabel("Difference with respect to baseline", fontsize = 16)

fig.colorbar(surf, shrink = 1)

plt.yticks( np.arange(0,30,3), np.arange(1,31,3))    
plt.xticks( np.arange(0,nr_experiments,4), np.arange(51,51+nr_experiments,4))   

plt.savefig(RESULTS_FOLDER + "\\" + "MEAN_3D_DIFFERENCE",bbox_inches='tight')
plt.close()
       









