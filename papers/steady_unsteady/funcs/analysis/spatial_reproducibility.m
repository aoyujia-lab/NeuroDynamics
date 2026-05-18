function [mean_r, sem_r, p_val, t_val, pair_corr_subj] = spatial_reproducibility(psd_all)
% Subject-level cross-run spatial pattern reproducibility of PSD
%
% INPUT:
%   psd_all : [nFreq × nROI × nSubj × nRun]
%
% OUTPUT:
%   mean_r         : [nFreq × 1] mean reproducibility (Fisher z averaged, back-transformed to r)
%   sem_r          : [nFreq × 1] SEM across subjects (in z space, back-transformed to r)
%   p_val          : [nFreq × 1] p-value from one-sample t-test (z > 0)
%   t_val          : [nFreq × 1] t-statistic
%   pair_corr_subj : [nFreq × nPairs × nSubj] raw pairwise correlations per subject

[nFreq, ~, nSubj, nRun] = size(psd_all);

n_pairs = nRun * (nRun - 1) / 2;
pair_corr_subj = nan(nFreq, n_pairs, nSubj);

for s = 1:nSubj
    for f = 1:nFreq
        pattern_sf = squeeze(psd_all(f, :, s, :));   % [nROI × nRun]
        k = 0;
        for i = 1:nRun-1
            for j = i+1:nRun
                k = k + 1;
                pair_corr_subj(f, k, s) = corr(pattern_sf(:, i), pattern_sf(:, j));
            end
        end
    end
end

% Fisher z transform
z_subj = atanh(pair_corr_subj);   % [nFreq × nPairs × nSubj]

% Clamp extreme values (if r = ±1, z = ±Inf)
z_subj(isinf(z_subj)) = sign(z_subj(isinf(z_subj))) * 5;

% Average across pairs within subject (in z space)
z_per_subj = squeeze(mean(z_subj, 2, 'omitnan'));   % [nFreq × nSubj]

% Group-level statistics
mean_z = mean(z_per_subj, 2, 'omitnan');             % [nFreq × 1]
sem_z  = std(z_per_subj, 0, 2, 'omitnan') / sqrt(nSubj);

% One-sample t-test against zero (in z space)
p_val = nan(nFreq, 1);
t_val = nan(nFreq, 1);
for f = 1:nFreq
    x = z_per_subj(f, :);
    x = x(~isnan(x));
    if length(x) < 3
        continue;
    end
    [~, pp, ~, stats] = ttest(x);
    p_val(f) = pp;
    t_val(f) = stats.tstat;
end

% Back-transform mean and SEM bounds to r space
mean_r = tanh(mean_z);
sem_r  = tanh(mean_z + sem_z) - mean_r;   % approximate upper SEM in r space

end