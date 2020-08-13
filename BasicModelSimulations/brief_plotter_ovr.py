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



corr_val = np.arange(0.3 , 1.1 ,0.1)
OVR = [ 0.8683   ,  0.6683  ,  0.5220  ,  0.3538 , 0.2888    ,  0.2451  , 0.2104 ,  0.1722]



plt.figure(1, figsize=(10,8))
plt.plot(corr_val, OVR, 'o-')
plt.xticks(corr_val)
ax = plt.gca()
yleft, yright = plt.ylim()
ax.set_ylim([0,1.06])
plt.title("Transmission Quality for no DBS",  fontsize = 20)
plt.xlabel("Correlation", fontsize = 16)
plt.ylabel("Transmission Quality", fontsize = 16)
plt.savefig("tq_c1_nodbs_100trials" ,bbox_inches='tight')
