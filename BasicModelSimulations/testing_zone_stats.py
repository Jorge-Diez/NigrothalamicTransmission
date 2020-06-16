# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 18:32:28 2020

@author: Jaymann
"""

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import os
import warnings



aa = np.random.normal(0, 0.1, 100000)


plt.figure(1)
plt.title('normal 100000 values no deletions')
plt.hist(aa,  ec='black', bins = 60)


nr_keep = 25000
chosen_positions = np.random.choice(aa.size, nr_keep, replace='False')

plt.figure(2)
plt.title('normal with 75000 deleted places')
plt.hist(aa[chosen_positions],  ec='black', bins = 60)



###exponential

bb = np.random.exponential(1, 100000)


plt.figure(3)
plt.title('exponential 100000 values no deletions')
plt.hist(bb,  ec='black', bins = 60)



plt.figure(4)
plt.title('exponential with 75000 deleted places')
plt.hist(bb[chosen_positions],  ec='black', bins = 60)


###lambda

cc = np.random.poisson(10, 100000)

plt.figure(5)
plt.title('lambda 100000 values no deletions')
plt.hist(cc,  ec='black', bins = 100)



plt.figure(6)
plt.title('exponential with 75000 deleted places')
plt.hist(cc[chosen_positions],  ec='black', bins = 60)

