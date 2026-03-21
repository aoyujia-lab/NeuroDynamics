function [r, pval] = dcor_colwise_optimized(X, y)

    if isvector(y)
        y = y(:);
    else
        error('y must be a vector.');
    end

    [n, p] = size(X);
    if size(y,1) ~= n
        error('X and y must have the same number of rows.');
    end

    r    = zeros(p, 1);
    pval = ones(p, 1);
    nPerm = 1000;

    % 1. precompute centered distance matrix for y
    By = local_centered_dist(y);
    dvarY2 = mean(By(:) .* By(:));

    if dvarY2 <= eps
        r(:) = 0;
        pval(:) = 1;
        return;
    end

    parfor j = 1:p
        x = X(:, j);
        Bx = local_centered_dist(x);
        dvarX2 = mean(Bx(:) .* Bx(:));

        if dvarX2 <= eps
            r(j) = 0;
            pval(j) = 1;
            continue;
        end

        % observed dCov^2
        dcov2_XY = mean(Bx(:) .* By(:));
        dcov2_XY = max(dcov2_XY, 0);

        r(j) = sqrt(dcov2_XY / sqrt(dvarX2 * dvarY2));

        % permutation test
        dcov2_perm_all = zeros(nPerm, 1);
        for ip = 1:nPerm
            idx = randperm(n);
            By_perm = By(idx, idx);
            dcov2_perm_all(ip) = max(mean(Bx(:) .* By_perm(:)), 0);
        end

        pval(j) = (sum(dcov2_perm_all >= dcov2_XY) + 1) / (nPerm + 1);
    end
end


function A = local_centered_dist(x)
    x = x(:);
    D = abs(x - x.');
    rowMean = mean(D, 2);
    colMean = mean(D, 1);
    grandMean = mean(D(:));
    A = D - rowMean - colMean + grandMean;
end