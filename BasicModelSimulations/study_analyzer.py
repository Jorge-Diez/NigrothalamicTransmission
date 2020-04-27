#!/usr/bin/python

import os
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as sio

#folder where our results will be located
RESULTS_FOLDER = "graphic_results"


if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)
    

#getting the corresponding directory for the basic experiment with 50 hz firing rate for all neurons
pwd = os.getcwd()
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
plt.title("Baseline 30 Neurons 50 Hz")
plt.xlabel("Correlation Coefficient")
cbar.set_label('Agents per grid cell')
plt.show()

#save figure without the corresponding "white space" around it
plt.savefig(RESULTS_FOLDER + '\\baseline50hz',bbox_inches='tight')