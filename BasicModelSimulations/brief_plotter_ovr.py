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



corr_val = np.arange(0,1.1,0.1)
OVR = [ 1.0000  ,  1.0000  ,  1.0000  ,  0.8000  ,  0.6167  ,  0.5750  ,  0.3867 ,  0.2833  ,  0.2950 ,   0.1933,
    0.2400]



plt.figure(1, figsize=(10,8))
plt.plot(corr_val, OVR, 'o-')
plt.xticks(corr_val)
ax = plt.gca()
yleft, yright = plt.ylim()
ax.set_ylim([0,yright])
plt.title("Transmission Quality for 30 SNr neurons, 50 Hz",  fontsize = 20)
plt.xlabel("Correlation", fontsize = 16)
plt.ylabel("Transmission Quality", fontsize = 16)
plt.savefig("tq_baseline" ,bbox_inches='tight')
