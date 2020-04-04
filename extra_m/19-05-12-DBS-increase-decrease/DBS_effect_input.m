%% Date created 12.05.19 by M. Mohagheghi

% This script generated spike trains with MIP (that generates noisy
% transmission via rebound spikes) and investigates the impact of
% increase-nochange-decrease in the firing rate of each individual spike
% train on the amplitude distribution of the resulting spike train.
global T_vec
T_vec = 0:0.01:1000;
rate = 50;
corr = 0.5;
N = 30;

ch_prob = [0, 0.4, 0.6]; %change probability

rate_ch_inc = [0.5, -0.8, 0]; %firig rate changes

[~, spks] = MIP_imp_v4_beta(corr, N, rate, T_vec);
figure;
subplot(211)
cnt = histcounts(spks, T_vec);
histogram(cnt(cnt~=0), [1:30])

init_indx = 1;

spk_dbs = [];
for id = 1:length(ch_prob)
    next_indx = N*ch_prob(id);
    spk_tmp = spks(init_indx: init_indx+next_indx-1, :);
    spkout = change_spk(spk_tmp, rate_ch_inc(id)*rate);
    spk_dbs = [spk_dbs, spkout];
    init_indx = init_indx + next_indx;
end

cnt = histcounts(spk_dbs, T_vec);
subplot(212)
histogram(cnt(cnt~=0), [1:30])

function new_spktr = change_spk(spktr, change_rate)
    global T_vec
    orig_rate = size(spktr, 2);
    new_spktr = [];
    if change_rate < 0
        for r_id = 1:size(spktr, 1)
            perm_inds = randperm(orig_rate, abs(change_rate));
            spktr_tmp = spktr(r_id, :);
            spktr_tmp(perm_inds) = [];
            new_spktr = [new_spktr, spktr_tmp];
        end
    elseif change_rate > 0
        for r_id = 1:size(spktr, 1)
            perm_inds = randperm(orig_rate, change_rate);
            spktr_tmp = MIP_imp_v4_beta(0, 1, change_rate, T_vec);
            new_spktr = [new_spktr, spktr_tmp'];
        end
        new_spktr = [new_spktr, reshape(spktr, [1, numel(spktr)])];
    else
        new_spktr = reshape(spktr, [1, numel(spktr)]);
    end
end