function [R_obs, P_obs, sig_cluster, stat] = freq_corr_cluster_perm(Y, X, varargin)
% freq_corr_cluster_perm
%
% Correlate one subject-level variable with frequency-wise data,
% then perform 1D cluster-based permutation correction across frequency.
%
% INPUT
%   Y : 1 x N or N x 1 vector
%       Subject-level variable, e.g., behavior
%
%   X : nFreq x N matrix
%       Frequency-wise data. Each row is one frequency bin.
%       Each column is one subject.
%
% OPTIONAL PARAMETERS
%   'nPerm'               : number of permutations, default = 5000
%   'clusterFormingAlpha' : pointwise threshold for forming clusters, default = 0.05
%   'clusterAlpha'        : cluster-level significance threshold, default = 0.05
%   'corrType'            : 'Pearson' or 'Spearman', default = 'Pearson'
%   'rngSeed'             : random seed, default = []
%
% OUTPUT
%   R_obs       : nFreq x 1 observed correlation values
%   P_obs       : nFreq x 1 pointwise p-values
%   sig_cluster : nFreq x 1 logical mask for cluster-corrected significant bins
%   stat        : structure containing detailed results

% -------------------------
% Parse inputs
% -------------------------

p = inputParser;

addParameter(p, 'nPerm', 5000, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'clusterFormingAlpha', 0.05, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
addParameter(p, 'clusterAlpha', 0.05, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
addParameter(p, 'corrType', 'Pearson', @(x) ischar(x) || isstring(x));
addParameter(p, 'rngSeed', [], @(x) isempty(x) || isnumeric(x));

parse(p, varargin{:});

nPerm = p.Results.nPerm;
clusterFormingAlpha = p.Results.clusterFormingAlpha;
clusterAlpha = p.Results.clusterAlpha;
corrType = char(p.Results.corrType);
rngSeed = p.Results.rngSeed;

if ~isempty(rngSeed)
    rng(rngSeed);
end

% -------------------------
% Check dimensions
% -------------------------

Y = Y(:)';  % force row vector

[nFreq, N] = size(X);

if length(Y) ~= N
    error('Dimension mismatch: Y must have the same number of subjects as columns of X.');
end

if any(isnan(Y)) || any(isnan(X(:)))
    error('This function currently does not allow NaN values. Please remove or impute missing data first.');
end

df = N - 2;

% -------------------------
% Spearman option
% -------------------------

switch lower(corrType)
    case 'pearson'
        Y_corr = Y;
        X_corr = X;

    case 'spearman'
        Y_corr = tiedrank(Y);

        X_corr = zeros(size(X));
        for f = 1:nFreq
            X_corr(f, :) = tiedrank(X(f, :));
        end

    otherwise
        error('corrType must be either Pearson or Spearman.');
end

% -------------------------
% Observed correlation
% -------------------------

[R_obs, P_obs, T_obs] = rowwise_corr(X_corr, Y_corr);

% two-tailed cluster-forming threshold
tcrit = tinv(1 - clusterFormingAlpha / 2, df);

sig_pos_obs = T_obs > tcrit;
sig_neg_obs = T_obs < -tcrit;

pos_clusters = find_1d_clusters(sig_pos_obs, T_obs, 'pos');
neg_clusters = find_1d_clusters(sig_neg_obs, T_obs, 'neg');

% -------------------------
% Permutation null distribution
% -------------------------

max_cluster_mass_null = zeros(nPerm, 1);

for iperm = 1:nPerm

    Y_perm = Y_corr(randperm(N));

    [~, ~, T_perm] = rowwise_corr(X_corr, Y_perm);

    sig_pos_perm = T_perm > tcrit;
    sig_neg_perm = T_perm < -tcrit;

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

% -------------------------
% Cluster-level p-values
% -------------------------

for i = 1:length(pos_clusters)
    pos_clusters(i).p_cluster = ...
        (sum(max_cluster_mass_null >= pos_clusters(i).mass) + 1) / (nPerm + 1);
end

for i = 1:length(neg_clusters)
    neg_clusters(i).p_cluster = ...
        (sum(max_cluster_mass_null >= neg_clusters(i).mass) + 1) / (nPerm + 1);
end

% -------------------------
% Significant cluster mask
% -------------------------

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

% -------------------------
% Output structure
% -------------------------

stat = struct;

stat.R_obs = R_obs;
stat.P_obs = P_obs;
stat.T_obs = T_obs;

stat.tcrit = tcrit;
stat.df = df;

stat.pos_clusters = pos_clusters;
stat.neg_clusters = neg_clusters;
stat.max_cluster_mass_null = max_cluster_mass_null;

stat.pointwise_sig = P_obs < clusterFormingAlpha;
stat.sig_cluster = sig_cluster;

stat.settings.nPerm = nPerm;
stat.settings.clusterFormingAlpha = clusterFormingAlpha;
stat.settings.clusterAlpha = clusterAlpha;
stat.settings.corrType = corrType;

end


% ============================================================
% Helper function: row-wise correlation
% ============================================================

function [R, P, T] = rowwise_corr(X, Y)

[nFreq, N] = size(X);
df = N - 2;

Y = Y(:)';

X_centered = X - mean(X, 2);
Y_centered = Y - mean(Y);

numerator = X_centered * Y_centered';
denominator = sqrt(sum(X_centered.^2, 2) .* sum(Y_centered.^2));

R = numerator ./ denominator;

% Avoid numerical overflow when R is extremely close to +/-1
R = max(min(R, 1 - eps), -1 + eps);

T = R .* sqrt(df ./ (1 - R.^2));

P = 2 * tcdf(-abs(T), df);

R = R(:);
P = P(:);
T = T(:);

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
            cluster_mass = sum(T(idx));

        case 'neg'
            cluster_mass = sum(abs(T(idx)));

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