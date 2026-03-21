function P = permtest_predict_behavior_psd_elasticnet_beta(all_vects, all_behav, cov, C, nPerm)
% Permutation test for prediction + feature-level beta/significance
% Observed: use C as-is (e.g., repeat=10)
% Permutation: override repeat=1 for speed, but still compute beta summaries

if nargin < 3, cov = []; end
if nargin < 4, C = struct(); end
if nargin < 5 || isempty(nPerm), nPerm = 10000; end
if ~isfield(C,'ml'), C.ml = struct(); end

% ---------------- Observed (full settings) ----------------
Sobs = predict_behavior_psd_elasticnet(all_vects, all_behav, cov, C);

obs_r   = Sobs.eval.r_enet;
obs_mse = Sobs.eval.mse_enet;

% >>> add Q2 (requires predict_behavior_psd_elasticnet to output S.eval.Q2_enet)
if isfield(Sobs,'eval') && isfield(Sobs.eval,'Q2_enet')
    obs_Q2 = Sobs.eval.Q2_enet;
else
    error('Sobs.eval.Q2_enet not found. Please add Q2 output in predict_behavior_psd_elasticnet first.');
end

Bfold = Sobs.model.beta_perfold;              % [nFeat x nSubj]
obs_beta_mean = mean(Bfold, 2, 'omitnan');    % per-feature mean beta across folds
obs_sel_freq  = mean(Bfold ~= 0, 2, 'omitnan'); % per-feature selection frequency

nFeat = size(Bfold,1);
nSubj = numel(all_behav);

% ---------------- Permutation settings (speedup) ----------------
Cperm = C;
if ~isfield(Cperm,'ml') || ~isstruct(Cperm.ml), Cperm.ml = struct(); end
Cperm.ml.enet_repeat = 1;   % <<< key speedup in permutation
% keep everything else identical to preserve the null model class

% ---------------- Allocate null ----------------
null_r   = nan(nPerm,1);
null_mse = nan(nPerm,1);
null_Q2  = nan(nPerm,1);    % <<< add

null_beta_mean = nan(nFeat, nPerm);
null_sel_freq  = nan(nFeat, nPerm);
null_max_abs_beta_mean = nan(nPerm,1);

seed0 = 1;
rng(seed0,'twister');

% ---------------- Permutation loop ----------------
for pp = 1:nPerm
    pp
    perm_idx = randperm(nSubj);
    yperm = all_behav(perm_idx);

    Sper = predict_behavior_psd_elasticnet(all_vects, yperm, cov, Cperm);

    null_r(pp)   = Sper.eval.r_enet;
    null_mse(pp) = Sper.eval.mse_enet;

    % >>> add Q2
    if isfield(Sper,'eval') && isfield(Sper.eval,'Q2_enet')
        null_Q2(pp) = Sper.eval.Q2_enet;
    else
        error('Sper.eval.Q2_enet not found. Please add Q2 output in predict_behavior_psd_elasticnet first.');
    end

    Bp = Sper.model.beta_perfold;                 % [nFeat x nSubj]
    bm = mean(Bp, 2, 'omitnan');
    sf = mean(Bp ~= 0, 2, 'omitnan');

    null_beta_mean(:,pp) = bm;
    null_sel_freq(:,pp)  = sf;
    null_max_abs_beta_mean(pp) = max(abs(bm));
end

% ---------------- P-values ----------------
% prediction
p_r   = (sum(abs(null_r) >= abs(obs_r)) + 1) / (nPerm + 1);      % two-sided
p_mse = (sum(null_mse <= obs_mse) + 1) / (nPerm + 1);            % one-sided (smaller better)

% >>> Q2 p-value (one-sided: larger Q2 is better)
p_Q2  = (sum(null_Q2 >= obs_Q2) + 1) / (nPerm + 1);

% feature-level (uncorrected)
p_beta_mean_2s = (sum(abs(null_beta_mean) >= abs(obs_beta_mean), 2) + 1) / (nPerm + 1);
p_sel_freq_1s  = (sum(null_sel_freq >= obs_sel_freq, 2) + 1) / (nPerm + 1);

% FWE via max-statistic on |beta_mean|
p_beta_mean_fwe = (sum(null_max_abs_beta_mean >= abs(obs_beta_mean)', 1) + 1) / (nPerm + 1);
p_beta_mean_fwe = p_beta_mean_fwe(:);

% ---------------- Pack ----------------
P = struct();
P.obs = struct();
P.obs.S = Sobs;
P.obs.r = obs_r;
P.obs.mse = obs_mse;
P.obs.Q2  = obs_Q2;              % <<< add
P.obs.beta_mean = obs_beta_mean;
P.obs.sel_freq  = obs_sel_freq;

P.null = struct();
P.null.r = null_r;
P.null.mse = null_mse;
P.null.Q2  = null_Q2;            % <<< add
P.null.beta_mean = null_beta_mean;
P.null.sel_freq  = null_sel_freq;
P.null.max_abs_beta_mean = null_max_abs_beta_mean;

P.p = struct();
P.p.r = p_r;
P.p.mse = p_mse;
P.p.Q2  = p_Q2;                  % <<< add
P.p.beta_mean_2s = p_beta_mean_2s;      % raw (uncorrected) permutation p
P.p.sel_freq_1s  = p_sel_freq_1s;       % raw (uncorrected) permutation p
P.p.beta_mean_fwe = p_beta_mean_fwe;    % FWE-corrected

P.meta = struct('nPerm',nPerm,'seed',seed0,'perm_repeat',Cperm.ml.enet_repeat,'obs_repeat',C.ml.enet_repeat);

end