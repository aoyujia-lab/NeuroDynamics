function [r, pval] = dcor_colwise(X, y, n_perm)
%DCOR_COLWISE Distance correlation for each column of X against y.
%
%   [r, pval] = dcor_colwise(X, y)
%   [r, pval] = dcor_colwise(X, y, n_perm)

if nargin < 3 || isempty(n_perm)
    n_perm = 1000;
end

if ~isvector(y)
    error('y must be a vector.');
end

y = y(:);
[n, n_feature] = size(X);
if size(y, 1) ~= n
    error('X and y must have the same number of rows.');
end

r = zeros(n_feature, 1);
pval = ones(n_feature, 1);

By = centered_distance_matrix(y);
dvar_y = mean(By(:) .^ 2);
if dvar_y <= eps
    return;
end

parfor j = 1:n_feature
    Bx = centered_distance_matrix(X(:, j));
    dvar_x = mean(Bx(:) .^ 2);
    if dvar_x <= eps
        continue;
    end

    dcov_xy = max(mean(Bx(:) .* By(:)), 0);
    r(j) = sqrt(dcov_xy / sqrt(dvar_x * dvar_y));

    dcov_perm = zeros(n_perm, 1);
    for ip = 1:n_perm
        perm_idx = randperm(n);
        By_perm = By(perm_idx, perm_idx);
        dcov_perm(ip) = max(mean(Bx(:) .* By_perm(:)), 0);
    end

    pval(j) = (sum(dcov_perm >= dcov_xy) + 1) / (n_perm + 1);
end
end

function A = centered_distance_matrix(x)
x = x(:);
D = abs(x - x.');
A = D - mean(D, 2) - mean(D, 1) + mean(D(:));
end
