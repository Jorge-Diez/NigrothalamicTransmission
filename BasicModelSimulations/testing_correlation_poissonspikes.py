import numpy as np
import matplotlib.pyplot as plt


#script made to test correlation between mother spike trains, and if it mean should be 0


#generate the first "mother" spike train

#time is kept in seconds, this explains the /1000 divisions


def poisson_spiketrains (T, Firing_rate, nr_spiketrains):
    
    dt = (T[1] - T[0]) / 1000
    prob = Firing_rate * dt
    U = np.random.uniform(size=(nr_spiketrains, T.size))
    spk = np.zeros((nr_spiketrains, T.size))
    spk[U<=prob] = 1
    
    return spk




# FIRST WITH NO CORRELATION WHATSOEVER, JUST "MOTHER SPIKE TRAINS"
T = np.arange(0,1000.01,0.01)

F_1 = 50
F_2 = 80


baseline = poisson_spiketrains(T, F_1, 15)  #15 MOTHER SPIKE TRAINS AT BASELINE
augmented = poisson_spiketrains(T, F_2, 15) # 15 INCREASED SPIKE TRAINS

spike_bits = np.concatenate((baseline, augmented))
nr_neurons = 15

corr_bin = float(input("select bin in ms for correlation matrix : "))
jumps = int(100000 / len(np.arange(0,1000,corr_bin)))  #used later  

spike_bits_binned = np.zeros( (30, len(np.arange(0,1000,corr_bin))))

for i in range(len(np.arange(0,1000,corr_bin))-1):
    all_spike_bits_slice = spike_bits[:,i*jumps:(i+1)*jumps]
    sliced_spike_bits = np.sum(all_spike_bits_slice, axis=1)
    
    spike_bits_binned[: ,i ] = sliced_spike_bits
    
    
    
 #since time goes from 0-1000 (1000 included, we add the last row to the last bin)
i += 1
all_spike_bits_slice = spike_bits[:,i*jumps:spike_bits.shape[1]-1]
sliced_spike_bits = np.sum(all_spike_bits_slice, axis=1)
spike_bits_binned[: ,i ] = sliced_spike_bits





corr_matrix = np.corrcoef(spike_bits_binned)
upper_mask =  np.tri(corr_matrix.shape[0], k=-1)
upper_mask[np.diag_indices(upper_mask.shape[0])] = 1


upper_corr_matrix = np.ma.array(corr_matrix, mask=upper_mask)

#following calculations made for means
zero_mask = np.ma.array(corr_matrix, mask=upper_mask, fill_value=0 )
masked_zeroed_corr_matrix = zero_mask.filled()
#calculate means
full_mean= np.sum(masked_zeroed_corr_matrix)/ np.count_nonzero(masked_zeroed_corr_matrix)


plt.figure(1, figsize=(10,8))
plt.matshow(upper_corr_matrix, fignum=9)
plt.plot(  [0,30] , [30-nr_neurons-0.5,30-nr_neurons-0.5], 'k'   )
plt.plot(  [30-nr_neurons-0.5,30-nr_neurons-0.5] , [-0.5,30] , 'k'  )
print("MEAN IS: " + str(full_mean))
plt.colorbar()
plt.show()


























