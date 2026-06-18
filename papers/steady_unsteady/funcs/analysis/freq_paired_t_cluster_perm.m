function [T_obs, P_obs, sig_cluster, stat] = freq_paired_t_cluster_perm(X, Y, varargin)
% freq_paired_t_cluster_perm
%
% Perform frequency-wise paired t-tests between two nFreq x N matrices,
% then perform 1D cluster-based permutation correction across frequency.
%
% INPUT
%   X : nFreq x N matrix
%       Frequency-wise data from condition X.
%       Each row is one frequency bin.
%       Each column is one subject.
%
%   Y : nFreq x N matrix
%       Frequency-wise data from condition Y.
%       Must have the same size as X.
%
% OPTIONAL PARAMETERS
%   'nPerm'               : number of permutations, default = 5000
%   'clusterFormingAlpha' : pointwise threshold for forming clusters, default = 0.05
%   'clusterAlpha'        : cluster-level significance threshold, default = 0.05
%   'rngSeed'             : random seed, default = []
%
% OUTPUT
%   T_obs       : nFreq x 1 observed t values
%   P_obs       : nFreq x 1 pointwise p-values
%   sig_cluster : nFreq x 1 logical mask for cluster-corrected significant bins
%   stat        : structure containing detailed results
%
% NOTE
%   The test is based on D = X - Y.
%   Therefore, positive T values indicate X > Y.

% -------------------------
% Parse inputs
% -------------------------

p = inputParser;

addParameter(p, 'nPerm', 5000, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'clusterFormingAlpha', 0.05, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
addParameter(p, 'clusterAlpha', 0.05, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
addParameter(p, 'rngSeed', [], @(x) isempty(x) || isnumeric(x));

parse(p, varargin{:});

nPerm = p.Results.nPerm;
clusterFormingAlpha = p.Results.clusterFormingAlpha;
clusterAlpha = p.Results.clusterAlpha;
rngSeed = p.Results.rngSeed;

if ~isempty(rngSeed)
    rng(rngSeed);
end

% -------------------------
% Check dimensions
% -------------------------

if ~ismatrix(X) || ~ismatrix(Y)
    error('X and Y must both be 2D matrices.');
end

if ~isequal(size(X), size(Y))
    error('Dimension mismatch: X and Y must have the same size.');
end

[nFreq, N] = size(X);

if N < 2
    error('At least two subjects are required for a paired t-test.');
end

if any(isnan(X(:))) || any(isnan(Y(:)))
    error('This function currently does not allow NaN values. Please remove or impute missing data first.');
end

df = N - 1;

% -------------------------
% Observed paired t-test
% -------------------------

D_obs = X - Y;

[T_obs, P_obs, mean_diff, sd_diff] = rowwise_paired_t(D_obs);

% two-tailed cluster-forming threshold
tcrit = tinv(1 - clusterFormingAlpha / 2, df);

sig_pos_obs = T_obs > tcrit;
sig_neg_obs = T_obs < -tcrit;

pos_clusters = find_1d_clusters(sig_pos_obs, T_obs, 'pos');
neg_clusters = find_1d_clusters(sig_neg_obs, T_obs, 'neg');

% -------------------------
% Permutation null distribution
% -------------------------
%
% For paired data, the permutation is performed by randomly flipping
% the sign of X - Y for each subject.

max_cluster_mass_null = zeros(nPerm, 1);

for iperm = 1:nPerm

    flip_sign = rand(1, N) > 0.5;
    flip_sign = double(flip_sign) * 2 - 1;  % convert to -1 or +1

    D_perm = bsxfun(@times, D_obs, flip_sign);

    [T_perm, ~] = rowwise_paired_t(D_perm);

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

stat.T_obs = T_obs;
stat.P_obs = P_obs;

stat.mean_diff = mean_diff;
stat.sd_diff = sd_diff;

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
stat.settings.test = 'paired t-test';
stat.settings.permutation = 'sign-flipping';
stat.settings.contrast = 'X - Y';

end


% ============================================================
% Helper function: row-wise paired t-test
% ============================================================

function [T, P, mean_diff, sd_diff] = rowwise_paired_t(D)

[nFreq, N] = size(D);
df = N - 1;

mean_diff = mean(D, 2);
sd_diff = std(D, 0, 2);

se_diff = sd_diff ./ sqrt(N);

T = mean_diff ./ se_diff;

% Handle zero-variance cases explicitly
zero_var = sd_diff == 0;

T(zero_var & mean_diff == 0) = 0;
T(zero_var & mean_diff > 0) = Inf;
T(zero_var & mean_diff < 0) = -Inf;

P = 2 * tcdf(-abs(T), df);

P(T == 0 & zero_var) = 1;

T = T(:);
P = P(:);
mean_diff = mean_diff(:);
sd_diff = sd_diff(:);

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