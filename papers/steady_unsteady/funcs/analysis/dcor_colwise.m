function [r, pval] = dcor_colwise(X, y, n_perm)
%DCOR_COLWISE Distance correlation for each column of X against y.
%
%   r = dcor_colwise(X, y)
%       Only compute distance correlation. No significance test.
%
%   [r, pval] = dcor_colwise(X, y)
%       Only compute distance correlation. pval will be NaN.
%
%   [r, pval] = dcor_colwise(X, y, n_perm)
%       Compute distance correlation and permutation-based p-values.
%
% INPUT:
%   X      : [nSample x nFeature] or [nSample x 1]
%   y      : [nSample x 1]
%   n_perm : number of permutations. If omitted or empty, no test.
%
% OUTPUT:
%   r      : [nFeature x 1] distance correlation
%   pval   : [nFeature x 1] permutation p-value, or NaN if not tested

if nargin < 3
    n_perm = [];
end

do_perm = ~isempty(n_perm) && n_perm > 0;

if ~isvector(y)
    error('y must be a vector.');
end

y = y(:);

if isvector(X)
    X = X(:);
end

[n, n_feature] = size(X);

if size(y, 1) ~= n
    error('X and y must have the same number of rows.');
end

r = zeros(n_feature, 1);
pval = nan(n_feature, 1);

By = centered_distance_matrix(y);
dvar_y = mean(By(:) .^ 2);

if dvar_y <= eps
    return;
end

for j = 1:n_feature
    xj = X(:, j);

    Bx = centered_distance_matrix(xj);
    dvar_x = mean(Bx(:) .^ 2);

    if dvar_x <= eps
        continue;
    end

    dcov_xy = max(mean(Bx(:) .* By(:)), 0);

    r(j) = sqrt(dcov_xy / sqrt(dvar_x * dvar_y));

    if do_perm
        dcov_perm = zeros(n_perm, 1);

        for ip = 1:n_perm
            perm_idx = randperm(n);
            By_perm = By(perm_idx, perm_idx);
            dcov_perm(ip) = max(mean(Bx(:) .* By_perm(:)), 0);
        end

        pval(j) = (sum(dcov_perm >= dcov_xy) + 1) / (n_perm + 1);
    end
end

end


function A = centered_distance_matrix(x)
x = x(:);
D = abs(x - x.');
A = D - mean(D, 2) - mean(D, 1) + mean(D(:));
end