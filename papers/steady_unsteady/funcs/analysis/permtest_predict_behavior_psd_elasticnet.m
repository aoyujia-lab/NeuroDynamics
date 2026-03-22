function P = permtest_predict_behavior_psd_elasticnet_beta(all_vects, all_behav, cov, C)
% Permutation test for prediction + feature-level beta/significance
% Memory-optimized version:
% - observed run uses original C
% - permutation run forces enet_repeat = 1
% - feature-level p-values are accumulated online, without storing full
%   [nFeat x nPerm] null matrices

if nargin < 3 || isempty(cov)
    cov = [];
end
if nargin < 4 || isempty(C)
    C = struct();
end
if ~isfield(C, 'ml') || ~isstruct(C.ml)
    C.ml = struct();
end

nPerm = C.stats.n_perm;
nSubj = numel(all_behav);

% ==================== Observed ====================
Sobs = predict_behavior_psd_elasticnet(all_vects, all_behav, cov, C);

obs_r   = Sobs.eval.r_enet;
obs_mse = Sobs.eval.mse_enet;
obs_Q2  = Sobs.eval.Q2_enet;

Bfold = Sobs.model.beta_perfold;                  % [nFeat x nSubj]
obs_beta_mean = mean(Bfold, 2, 'omitnan');
obs_sel_freq  = mean(Bfold ~= 0, 2, 'omitnan');

nFeat = size(Bfold, 1);

% ==================== Permutation settings ====================
Cperm = C;
if ~isfield(Cperm, 'ml') || ~isstruct(Cperm.ml)
    Cperm.ml = struct();
end
Cperm.ml.enet_repeat = 1;

% ==================== Allocate null (scalar metrics only) ====================
null_r   = nan(nPerm, 1);
null_mse = nan(nPerm, 1);
null_Q2  = nan(nPerm, 1);
null_max_abs_beta_mean = nan(nPerm, 1);

% online counters for feature-level p-values
count_beta_mean_2s = zeros(nFeat, 1);
count_sel_freq_1s  = zeros(nFeat, 1);
count_beta_mean_fwe = zeros(nFeat, 1);

seed0 = 1;
rng(seed0, 'twister');

obs_abs_beta_mean = abs(obs_beta_mean);

% ==================== Permutation loop ====================
for pp = 1:nPerm
    perm_idx = randperm(nSubj);
    yperm = all_behav(perm_idx);

    Sper = predict_behavior_psd_elasticnet(all_vects, yperm, cov, Cperm);

    null_r(pp)   = Sper.eval.r_enet;
    null_mse(pp) = Sper.eval.mse_enet;
    null_Q2(pp)  = Sper.eval.Q2_enet;

    Bp = Sper.model.beta_perfold;                 % [nFeat x nSubj]
    bm = mean(Bp, 2, 'omitnan');
    sf = mean(Bp ~= 0, 2, 'omitnan');

    abs_bm = abs(bm);
    max_abs_bm = max(abs_bm);

    null_max_abs_beta_mean(pp) = max_abs_bm;

    % online accumulation
    count_beta_mean_2s = count_beta_mean_2s + (abs_bm >= obs_abs_beta_mean);
    count_sel_freq_1s  = count_sel_freq_1s  + (sf >= obs_sel_freq);
    count_beta_mean_fwe = count_beta_mean_fwe + (max_abs_bm >= obs_abs_beta_mean);

    if mod(pp, 50) == 0 || pp == nPerm
        fprintf('Permutation %d/%d\n', pp, nPerm);
    end
end

% ==================== P-values ====================
p_r   = (sum(abs(null_r) >= abs(obs_r)) + 1) / (nPerm + 1);
p_mse = (sum(null_mse <= obs_mse) + 1) / (nPerm + 1);
p_Q2  = (sum(null_Q2 >= obs_Q2) + 1) / (nPerm + 1);

p_beta_mean_2s = (count_beta_mean_2s + 1) / (nPerm + 1);
p_sel_freq_1s  = (count_sel_freq_1s  + 1) / (nPerm + 1);
p_beta_mean_fwe = (count_beta_mean_fwe + 1) / (nPerm + 1);

% ==================== Pack ====================
P = struct();

P.obs = struct();
P.obs.S = Sobs;
P.obs.r = obs_r;
P.obs.mse = obs_mse;
P.obs.Q2 = obs_Q2;
P.obs.beta_mean = obs_beta_mean;
P.obs.sel_freq = obs_sel_freq;

P.null = struct();
P.null.r = null_r;
P.null.mse = null_mse;
P.null.Q2 = null_Q2;
P.null.max_abs_beta_mean = null_max_abs_beta_mean;

P.p = struct();
P.p.r = p_r;
P.p.mse = p_mse;
P.p.Q2 = p_Q2;
P.p.beta_mean_2s = p_beta_mean_2s;
P.p.sel_freq_1s = p_sel_freq_1s;
P.p.beta_mean_fwe = p_beta_mean_fwe;

P.meta = struct( ...
    'nPerm', nPerm, ...
    'seed', seed0, ...
    'perm_repeat', Cperm.ml.enet_repeat, ...
    'obs_repeat', C.ml.enet_repeat, ...
    'store_full_feature_null', false);

end