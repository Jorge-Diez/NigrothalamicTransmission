function [freqs, nr_neurons] = par_obtain(nr_neurons_increase, freq_increase, N_SNr)

if (nr_neurons_increase == N_SNr)
    freqs = freq_increase; %Hz
    
    nr_neurons = nr_neurons_increase; %groups of neurons with specific firing rates
    
else
    
    %obtain parameters
    freqs = [50 freq_increase]; %Hz
    
    nr_neurons = [N_SNr - nr_neurons_increase nr_neurons_increase]; %groups of neurons with specific firing rates
end


end