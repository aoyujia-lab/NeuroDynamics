function stat = psd_sliding_corr_allfreq(psd_out, ranges, varargin)
%PSD_SLIDING_CORR_ALLFREQ
%
% Compute Pearson correlation and regression beta between:
%   X(t)   = whole-brain mean power in ranges{1}+ranges{2}
%   Y_f(t) = whole-brain mean power at each frequency point f
%
% across sliding windows.
%
% Then perform group-level 1D cluster-based permutation correction
% across frequency using Fisher-z transformed correlation values.
%
% INPUT
%   psd_out : [nFreq x nROI x nWin x nSubj x nSes]
%             or [nFreq x nROI x nWin x nSubj]
%
%   ranges  : cell
%             ranges{1}, ranges{2}: frequency bins used to construct X
%
% OPTIONAL
%   'zscore'     : true/false, default = false
%                  If true, beta becomes standardized beta.
%                  In simple regression, standardized beta is equivalent
%                  to Pearson correlation.
%
%   'exclude_X'  : true/false, default = true
%                  Whether to exclude ranges{1}+ranges{2} from target frequencies.
%
%   'target_idx' : target frequency indices, default = all non-X frequencies.
%
%   'store_ts'   : true/false, default = false
%
%   'nPerm'               : number of permutations, default = 5000
%   'clusterFormingAlpha' : pointwise threshold for forming clusters, default = 0.05
%   'clusterAlpha'        : cluster-level significance threshold, default = 0.05
%   'rngSeed'             : random seed, default = []
%
% OUTPUT
%   stat.r              : [nFreq x nSubj x nSes]
%                         individual/session-level Pearson r
%
%   stat.p              : [nFreq x nSubj x nSes]
%                         individual/session-level correlation p
%
%   stat.beta           : [nFreq x nSubj x nSes]
%                         individual/session-level regression slope
%                         Y_f = beta0 + beta * X
%
%   stat.beta0          : [nFreq x nSubj x nSes]
%                         individual/session-level intercept
%
%   stat.group_r_mean   : [nFreq x 1]
%                         Fisher-z averaged group mean r
%
%   stat.group_z_mean   : [nFreq x 1]
%                         group mean Fisher z
%
%   stat.group_t        : [nFreq x 1]
%                         group-level t value testing Fisher z against 0
%
%   stat.group_p        : [nFreq x 1]
%
%   stat.group_beta_mean : [nFreq x 1]
%                          group mean beta across subject/session observations
%
%   stat.group_beta_t    : [nFreq x 1]
%                          group-level t value testing beta against 0
%
%   stat.group_beta_p    : [nFreq x 1]
%
%   stat.group_beta_mean_subj : [nFreq x 1]
%                               beta averaged within subject first,
%                               then averaged across subjects
%
%   stat.sig_cluster    : [nFreq x 1]
%                         cluster-corrected significant frequency bins
%                         based on Fisher-z correlation t-values
%
%   stat.X_ts           : [nWin x nSubj x nSes]
%
%   stat.Y_ts           : optional, [nWin x nFreq x nSubj x nSes]

%% ---- Optional parameters ----
p = inputParser;

addParameter(p, 'zscore', false);
addParameter(p, 'exclude_X', true);
addParameter(p, 'target_idx', []);
addParameter(p, 'store_ts', false);

addParameter(p, 'nPerm', 5000, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'clusterFormingAlpha', 0.05, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
addParameter(p, 'clusterAlpha', 0.05, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
addParameter(p, 'rngSeed', [], @(x) isempty(x) || isnumeric(x));

parse(p, varargin{:});

do_zscore = p.Results.zscore;
exclude_X = p.Results.exclude_X;
target_idx_input = p.Results.target_idx;
store_ts = p.Results.store_ts;

nPerm = p.Results.nPerm;
clusterFormingAlpha = p.Results.clusterFormingAlpha;
clusterAlpha = p.Results.clusterAlpha;
rngSeed = p.Results.rngSeed;

if ~isempty(rngSeed)
    rng(rngSeed);
end

%% ---- Make sure psd_out is 5D ----
if ndims(psd_out) == 4
    psd_out = reshape(psd_out, ...
        size(psd_out, 1), size(psd_out, 2), size(psd_out, 3), size(psd_out, 4), 1);
end

[nFreq, ~, nWin, nSubj, nSes] = size(psd_out);

%% ---- Predictor frequency indices ----
idx_X = unique([ranges{1}(:); ranges{2}(:)]);
idx_X = ranges{4}(:);

if max(idx_X) > nFreq
    error('ranges{1} or ranges{2} exceeds nFreq.');
end

%% ---- Target frequency indices ----
if isempty(target_idx_input)
    target_idx = 1:nFreq;
else
    target_idx = target_idx_input(:).';
end

if exclude_X
    target_idx = setdiff(target_idx, idx_X);
end

if max(target_idx) > nFreq
    error('target_idx exceeds nFreq.');
end

%% ---- Preallocate ----
R = nan(nFreq, nSubj, nSes);
P = nan(nFreq, nSubj, nSes);

BETA = nan(nFreq, nSubj, nSes);
BETA0 = nan(nFreq, nSubj, nSes);

X_ts = nan(nWin, nSubj, nSes);

if store_ts
    Y_ts = nan(nWin, nFreq, nSubj, nSes);
else
    Y_ts = [];
end

%% ---- Individual-level correlation and beta ----
for isubj = 1:nSubj

    fprintf('Subject %d/%d\n', isubj, nSubj);

    for ises = 1:nSes

        this_psd = psd_out(:, :, :, isubj, ises);  % [nFreq x nROI x nWin]

        if all(isnan(this_psd(:)))
            continue
        end

        %% ---- X: ranges{1}+ranges{2}, whole-brain average ----
        X = squeeze(mean(sum(this_psd(idx_X, :, :), 1, 'omitnan'), 2, 'omitnan'));
        X = X(:);

        X_ts(:, isubj, ises) = X;

        %% ---- Correlate and regress X with each target frequency ----
        for ifreq = target_idx

            % Y: one frequency point, whole-brain average
            Y = squeeze(mean(this_psd(ifreq, :, :), 2, 'omitnan'));
            Y = Y(:);

            valid_idx = ~isnan(X) & ~isnan(Y);

            X_valid = X(valid_idx);
            Y_valid = Y(valid_idx);

            if numel(X_valid) < 4
                continue
            end

            if do_zscore
                if std(X_valid) <= eps || std(Y_valid) <= eps
                    continue
                end

                X_valid = zscore(X_valid);
                Y_valid = zscore(Y_valid);
            end

            if std(X_valid) <= eps || std(Y_valid) <= eps
                continue
            end

            %% ---- Pearson correlation ----
            [r_tmp, p_tmp] = corr(X_valid, Y_valid, ...
                'Type', 'Pearson', ...
                'Rows', 'complete');

            R(ifreq, isubj, ises) = r_tmp;
            P(ifreq, isubj, ises) = p_tmp;

            %% ---- Linear regression beta: Y = beta0 + beta * X ----
            Xc = X_valid - mean(X_valid);
            Yc = Y_valid - mean(Y_valid);

            beta_tmp = sum(Xc .* Yc) / sum(Xc .^ 2);
            beta0_tmp = mean(Y_valid) - beta_tmp * mean(X_valid);

            BETA(ifreq, isubj, ises) = beta_tmp;
            BETA0(ifreq, isubj, ises) = beta0_tmp;

            %% ---- Store time series if requested ----
            if store_ts
                tmp = nan(nWin, 1);
                tmp(valid_idx) = Y_valid;
                Y_ts(:, ifreq, isubj, ises) = tmp;
            end

        end
    end
end

%% ---- Fisher-z transform ----
R_clip = R;
R_clip(R_clip >= 1) = 1 - eps;
R_clip(R_clip <= -1) = -1 + eps;

Z = atanh(R_clip);  % [nFreq x nSubj x nSes]

Z_mat = reshape(Z, nFreq, nSubj * nSes);  % [nFreq x nObs]

%% ---- Group-level one-sample t-test on Fisher z ----
[group_z_mean, group_t, group_p, df_vec] = rowwise_onesample_t_nan(Z_mat);

group_r_mean = tanh(group_z_mean);

%% ---- Group-level beta mean and t-test ----
BETA_mat = reshape(BETA, nFreq, nSubj * nSes);  % [nFreq x nObs]

[group_beta_mean, group_beta_t, group_beta_p, group_beta_df] = ...
    rowwise_onesample_t_nan(BETA_mat);

%% ---- Subject-level beta mean, averaging sessions first ----
% This is more conservative if each subject has multiple sessions.
BETA_subj_mean = squeeze(mean(BETA, 3, 'omitnan'));  % [nFreq x nSubj]

[group_beta_mean_subj, group_beta_t_subj, group_beta_p_subj, group_beta_df_subj] = ...
    rowwise_onesample_t_nan(BETA_subj_mean);

%% ---- Cluster-forming threshold ----
tcrit_vec = nan(nFreq, 1);

for ifreq = target_idx
    if ~isnan(df_vec(ifreq)) && df_vec(ifreq) > 0
        tcrit_vec(ifreq) = tinv(1 - clusterFormingAlpha / 2, df_vec(ifreq));
    end
end

sig_pos_obs = false(nFreq, 1);
sig_neg_obs = false(nFreq, 1);

sig_pos_obs(target_idx) = group_t(target_idx) > tcrit_vec(target_idx);
sig_neg_obs(target_idx) = group_t(target_idx) < -tcrit_vec(target_idx);

pos_clusters = find_1d_clusters(sig_pos_obs, group_t, 'pos');
neg_clusters = find_1d_clusters(sig_neg_obs, group_t, 'neg');

%% ---- Permutation null distribution: sign flipping ----
max_cluster_mass_null = zeros(nPerm, 1);

nObs = size(Z_mat, 2);

for iperm = 1:nPerm

    signs = randi([0, 1], 1, nObs) * 2 - 1;  % +/- 1
    Z_perm = Z_mat .* signs;

    [~, T_perm, ~, ~] = rowwise_onesample_t_nan(Z_perm);

    sig_pos_perm = false(nFreq, 1);
    sig_neg_perm = false(nFreq, 1);

    sig_pos_perm(target_idx) = T_perm(target_idx) > tcrit_vec(target_idx);
    sig_neg_perm(target_idx) = T_perm(target_idx) < -tcrit_vec(target_idx);

    pos_perm_clusters = find_1d_clusters(sig_pos_perm, T_perm, 'pos');
    neg_perm_clusters = find_1d_clusters(sig_neg_perm, T_perm, 'neg');

    all_mass = [];

    if ~isempty(pos_perm_clusters)
        all_mass = [all_mass, [pos_perm_clusters.mass]];
    end

    if ~isempty(neg_perm_clusters)
        all_mass = [all_mass, [neg_perm_clusters.mass]];
    end

    if isempty(all_mass)
        max_cluster_mass_null(iperm) = 0;
    else
        max_cluster_mass_null(iperm) = max(all_mass);
    end
end

%% ---- Cluster-level p-values ----
for i = 1:length(pos_clusters)
    pos_clusters(i).p_cluster = ...
        (sum(max_cluster_mass_null >= pos_clusters(i).mass) + 1) / (nPerm + 1);
end

for i = 1:length(neg_clusters)
    neg_clusters(i).p_cluster = ...
        (sum(max_cluster_mass_null >= neg_clusters(i).mass) + 1) / (nPerm + 1);
end

%% ---- Significant cluster mask ----
sig_cluster = false(nFreq, 1);

for i = 1:length(pos_clusters)
    if pos_clusters(i).p_cluster < clusterAlpha
        sig_cluster(pos_clusters(i).idx) = true;
    end
end

for i = 1:length(neg_clusters)
    if neg_clusters(i).p_cluster < clusterAlpha
        sig_cluster(neg_clusters(i).idx) = true;
    end
end

%% ---- Output ----
stat = struct;

stat.r = R;
stat.p = P;
stat.z = Z;

stat.beta = BETA;
stat.beta0 = BETA0;

stat.group_r_mean = group_r_mean;
stat.group_z_mean = group_z_mean;
stat.group_t = group_t;
stat.group_p = group_p;

stat.group_beta_mean = group_beta_mean;
stat.group_beta_t = group_beta_t;
stat.group_beta_p = group_beta_p;
stat.group_beta_df = group_beta_df;

stat.group_beta_mean_subj = group_beta_mean_subj;
stat.group_beta_t_subj = group_beta_t_subj;
stat.group_beta_p_subj = group_beta_p_subj;
stat.group_beta_df_subj = group_beta_df_subj;

stat.tcrit = tcrit_vec;
stat.df = df_vec;

stat.pos_clusters = pos_clusters;
stat.neg_clusters = neg_clusters;
stat.max_cluster_mass_null = max_cluster_mass_null;

stat.pointwise_sig = group_p < clusterFormingAlpha;
stat.sig_cluster = sig_cluster;

stat.X_ts = X_ts;
stat.Y_ts = Y_ts;

stat.idx_X = idx_X;
stat.target_idx = target_idx;

stat.zscore = do_zscore;
stat.exclude_X = exclude_X;

stat.settings.nPerm = nPerm;
stat.settings.clusterFormingAlpha = clusterFormingAlpha;
stat.settings.clusterAlpha = clusterAlpha;
stat.settings.rngSeed = rngSeed;

end


% ============================================================
% Helper function: row-wise one-sample t-test allowing NaNs
% ============================================================

function [M, T, P, DF] = rowwise_onesample_t_nan(X)

% X: [nFreq x nObs]
[nFreq, ~] = size(X);

M = nan(nFreq, 1);
T = nan(nFreq, 1);
P = nan(nFreq, 1);
DF = nan(nFreq, 1);

for f = 1:nFreq

    x = X(f, :);
    x = x(~isnan(x));

    N = numel(x);

    if N < 3
        continue
    end

    M(f) = mean(x);
    s = std(x);

    if s <= eps
        continue
    end

    DF(f) = N - 1;
    T(f) = M(f) / (s / sqrt(N));
    P(f) = 2 * tcdf(-abs(T(f)), DF(f));

end

end


% ============================================================
% Helper function: find 1D clusters
% ============================================================

function clusters = find_1d_clusters(sig_vec, T, direction)

sig_vec = sig_vec(:);
T = T(:);

d = diff([false; sig_vec; false]);

start_idx = find(d == 1);
end_idx = find(d == -1) - 1;

clusters = struct( ...
    'idx', {}, ...
    'start_idx', {}, ...
    'end_idx', {}, ...
    'size', {}, ...
    'mass', {}, ...
    'direction', {}, ...
    'p_cluster', {} ...
    );

for i = 1:length(start_idx)

    idx = start_idx(i):end_idx(i);

    switch lower(direction)
        case 'pos'
            cluster_mass = sum(T(idx), 'omitnan');

        case 'neg'
            cluster_mass = sum(abs(T(idx)), 'omitnan');

        otherwise
            error('direction must be pos or neg.');
    end

    clusters(i).idx = idx;
    clusters(i).start_idx = start_idx(i);
    clusters(i).end_idx = end_idx(i);
    clusters(i).size = length(idx);
    clusters(i).mass = cluster_mass;
    clusters(i).direction = direction;
    clusters(i).p_cluster = NaN;

end

end