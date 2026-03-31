function stats = paired_permutation_ttest(x, y, nPerm)
% Paired permutation test using sign flipping.
%
% Inputs:
%   x, y  : paired data vectors
%   nPerm : number of permutations, e.g., 5000 or 10000
%
% Output:
%   stats.obs_diff   : observed mean difference (x - y)
%   stats.obs_t      : observed paired t statistic
%   stats.p          : two-sided permutation p-value
%   stats.perm_t     : null distribution of t statistics

    if nargin < 3
        nPerm = 5000;
    end

    x = x(:);
    y = y(:);

    valid = ~isnan(x) & ~isnan(y);
    x = x(valid);
    y = y(valid);

    d = x - y;
    n = numel(d);

    obs_diff = mean(d);
    obs_t = obs_diff / (std(d) / sqrt(n));

    perm_t = zeros(nPerm, 1);

    for iPerm = 1:nPerm
        signs = randi([0 1], n, 1) * 2 - 1;   % random +/-1
        d_perm = d .* signs;
        perm_t(iPerm) = mean(d_perm) / (std(d_perm) / sqrt(n));
    end

    p = (sum(abs(perm_t) >= abs(obs_t)) + 1) / (nPerm + 1);

    stats.obs_diff = obs_diff;
    stats.obs_t = obs_t;
    stats.p = p;
    stats.perm_t = perm_t;
    stats.n = n;
end