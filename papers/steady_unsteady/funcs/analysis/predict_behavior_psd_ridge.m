function S = predict_behavior_psd_ridge(all_vects, all_behav, cov, C)
% Ridge-regression control model for ROI-level features (PSD etc.)
%
% INPUT
%   all_vects : [nROI x nSubj]
%   all_behav : [nSubj x 1]
%   cov       : [nSubj x nCov] (kept for interface consistency; not used by ridge unless you extend it)
%   C         : config (expects C.ml.* fields; see "Ridge settings" below)
%
% OUTPUT (struct S)
%   S.meta.*, S.pred.*, S.eval.*, S.model.*, S.debug.*
%
% Notes:
% - Leave-one-out CV (LOO), z-score within each fold using TRAIN only (no leakage).
% - Ridge alpha (lambda) is chosen by inner CV on the TRAIN set (default 5-fold)
%   unless C.ml.ridge_alpha is provided as a positive scalar.

% -------------------- Input --------------------
assert(ismatrix(all_vects), 'all_vects must be [nROI x nSubj].');
nROI  = size(all_vects,1);
nSubj = size(all_vects,2);
assert(size(all_behav,1) == nSubj && size(all_behav,2) == 1, 'all_behav must be [nSubj x 1].');

% -------------------- Ridge settings --------------------
% Optional config fields (safe defaults):
%   C.ml.ridge_alpha      : fixed lambda (positive scalar). If absent/empty -> CV select.
%   C.ml.ridge_alphas     : candidate lambdas vector for CV selection (default logspace(-4,4,30))
%   C.ml.ridge_kfold      : inner CV folds (default 5)
%   C.ml.ridge_standardize: whether to z-score X within fold (default true; recommended)
%   C.ml.ridge_add_intercept: include intercept (default true; recommended)
%   C.ml.ridge_use_cov_as_features: append covariates as additional predictors (default false)

if isfield(C,'ml') && isfield(C.ml,'ridge_alpha')
    ridge_alpha_fixed = C.ml.ridge_alpha;
else
    ridge_alpha_fixed = [];
end

if isfield(C,'ml') && isfield(C.ml,'ridge_alphas') && ~isempty(C.ml.ridge_alphas)
    alpha_grid = C.ml.ridge_alphas(:);
else
    alpha_grid = logspace(-4,4,30)';  % candidate lambdas
end

if isfield(C,'ml') && isfield(C.ml,'ridge_kfold') && ~isempty(C.ml.ridge_kfold)
    Kfold = C.ml.ridge_kfold;
else
    Kfold = 5;
end

if isfield(C,'ml') && isfield(C.ml,'ridge_standardize')
    do_standardize = logical(C.ml.ridge_standardize);
else
    do_standardize = true;
end

if isfield(C,'ml') && isfield(C.ml,'ridge_add_intercept')
    add_intercept = logical(C.ml.ridge_add_intercept);
else
    add_intercept = true;
end

if isfield(C,'ml') && isfield(C.ml,'ridge_use_cov_as_features')
    use_cov = logical(C.ml.ridge_use_cov_as_features);
else
    use_cov = false;
end

if use_cov
    assert(~isempty(cov) && size(cov,1) == nSubj, 'cov must be [nSubj x nCov] when ridge_use_cov_as_features = true.');
    nCov = size(cov,2);
else
    nCov = 0;
end

% -------------------- Meta --------------------
S = struct();
S.meta = struct();
S.meta.nROI      = nROI;
S.meta.nSubj     = nSubj;
S.meta.note      = 'LOO ridge regression on ROI-level features (with inner-CV alpha selection unless fixed)';
S.meta.ridge = struct();
S.meta.ridge.fixed_alpha   = ridge_alpha_fixed;
S.meta.ridge.alpha_grid    = alpha_grid;
S.meta.ridge.kfold         = Kfold;
S.meta.ridge.standardize   = do_standardize;
S.meta.ridge.add_intercept = add_intercept;
S.meta.ridge.use_cov_as_features = use_cov;
S.meta.ridge.nCov = nCov;

fprintf('Ridge control: Leave-one-out CV (%d subjects)\n', nSubj);
if ~isempty(ridge_alpha_fixed) && isscalar(ridge_alpha_fixed) && ridge_alpha_fixed > 0
    fprintf('  alpha (lambda) fixed: %.6g\n', ridge_alpha_fixed);
else
    fprintf('  alpha (lambda) selected by inner %d-fold CV on each TRAIN fold\n', Kfold);
end
fprintf('  standardize (train-only): %d, intercept: %d, use_cov_as_features: %d\n', do_standardize, add_intercept, use_cov);

% -------------------- Allocate --------------------
pred_ridge = nan(nSubj,1);

% per-fold model params
alpha_sel  = nan(nSubj,1);
beta_all   = nan(nROI + nCov, nSubj);   % coefficients for standardized features (if standardized)
inter_all  = nan(nSubj,1);

% store standardization params (for ROI features only; cov optionally)
mu_all = nan(nROI + nCov, nSubj);
sd_all = nan(nROI + nCov, nSubj);

% -------------------- LOO loop --------------------
for leftout = 1:nSubj
    train_idx = true(nSubj,1);
    train_idx(leftout) = false;

    % X: [nTrain x p]
    Xtrain = all_vects(:, train_idx)';   % transpose to subjects x features
    ytrain = all_behav(train_idx);

    Xtest  = all_vects(:, leftout)';     % [1 x nROI]

    if use_cov
        Xtrain = [Xtrain, cov(train_idx,:)];
        Xtest  = [Xtest,  cov(leftout,:)];
    end

    % ---- standardize using TRAIN only ----
    if do_standardize
        mu = mean(Xtrain, 1, 'omitnan');
        sd = std(Xtrain, 0, 1, 'omitnan');
        sd(sd == 0 | isnan(sd)) = 1; % avoid divide-by-zero
        XtrainZ = (Xtrain - mu) ./ sd;
        XtestZ  = (Xtest  - mu) ./ sd;
    else
        mu = zeros(1, size(Xtrain,2));
        sd = ones(1,  size(Xtrain,2));
        XtrainZ = Xtrain;
        XtestZ  = Xtest;
    end

    % ---- choose alpha on TRAIN ----
    if ~isempty(ridge_alpha_fixed) && isscalar(ridge_alpha_fixed) && ridge_alpha_fixed > 0
        a = ridge_alpha_fixed;
    else
        a = select_ridge_alpha_innercv(XtrainZ, ytrain, alpha_grid, Kfold, add_intercept);
    end
    alpha_sel(leftout) = a;

    % ---- fit ridge on full TRAIN ----
    [beta, intercept] = fit_ridge_closed_form(XtrainZ, ytrain, a, add_intercept);

    beta_all(:, leftout)  = beta(:);
    inter_all(leftout)    = intercept;
    mu_all(:, leftout)    = mu(:);
    sd_all(:, leftout)    = sd(:);

    % ---- predict TEST ----
    pred_ridge(leftout) = XtestZ * beta + intercept;
end

% -------------------- Evaluation --------------------
[r_ridge, p_ridge] = corr(pred_ridge, all_behav, 'rows', 'pairwise');

se_ridge  = (pred_ridge - all_behav).^2;
mse_ridge = mean(se_ridge, 'omitnan');

% optional calibration (same style as your CPM code)
cof_ridge = regress(all_behav, [pred_ridge, ones(nSubj,1)]);

% -------------------- Pack outputs --------------------
S.pred = struct();
% keep same field names style (pos/neg/com) by providing ridge as an extra field
S.pred.ridge = pred_ridge;

S.eval = struct();
S.eval.r_ridge   = r_ridge;
S.eval.p_ridge   = p_ridge;
S.eval.mse_ridge = mse_ridge;
S.eval.se_ridge  = se_ridge;
S.eval.calib_ridge = cof_ridge;

S.model = struct();
S.model.alpha_perfold     = alpha_sel;
S.model.beta_perfold      = beta_all;   % coefficients in standardized feature space (if standardized)
S.model.intercept_perfold = inter_all;

S.debug = struct();
S.debug.mu_perfold = mu_all;
S.debug.sd_perfold = sd_all;
S.debug.featureType = 'ROI-level';
S.debug.nFeat = nROI + nCov;

end


% ====================== helper: inner-CV alpha selection ======================
function best_alpha = select_ridge_alpha_innercv(X, y, alpha_grid, Kfold, add_intercept)
% X: [n x p] already standardized (train-only) at outer fold
% y: [n x 1]
n = size(X,1);

% handle tiny n
K = min(Kfold, max(2, n));  % at least 2-fold
cv_id = make_kfold_ids(n, K);

mse_grid = nan(numel(alpha_grid),1);

for ia = 1:numel(alpha_grid)
    a = alpha_grid(ia);
    se_all = nan(n,1);

    for k = 1:K
        te = (cv_id == k);
        tr = ~te;

        Xtr = X(tr,:);
        ytr = y(tr);
        Xte = X(te,:);
        yte = y(te);

        [beta, intercept] = fit_ridge_closed_form(Xtr, ytr, a, add_intercept);
        yhat = Xte * beta + intercept;

        se_all(te) = (yhat - yte).^2;
    end

    mse_grid(ia) = mean(se_all, 'omitnan');
end

% pick smallest MSE (ties -> smaller alpha for less shrinkage bias)
[~, ix] = min(mse_grid);
best_alpha = alpha_grid(ix);

end


% ====================== helper: ridge closed-form ======================
function [beta, intercept] = fit_ridge_closed_form(X, y, alpha, add_intercept)
% Solve ridge: min ||y - (X beta + intercept)||^2 + alpha ||beta||^2
% If add_intercept: center y and use intercept = mean(y) - mean(X)*beta
% NOTE: X should be standardized already if you want stable alpha scale.

[n, p] = size(X);

if add_intercept
    ybar = mean(y, 'omitnan');
    xbar = mean(X, 1, 'omitnan');
    yc = y - ybar;
    Xc = X - xbar;
else
    ybar = 0;
    xbar = zeros(1,p);
    yc = y;
    Xc = X;
end

% ridge closed form: beta = (X'X + alpha I)^(-1) X' y
A = (Xc' * Xc) + alpha * eye(p);
b = (Xc' * yc);
beta = A \ b;

if add_intercept
    intercept = ybar - xbar * beta;
else
    intercept = 0;
end

end


% ====================== helper: k-fold ids ======================
function cv_id = make_kfold_ids(n, K)
% deterministic-ish fold assignment
% If you want reproducibility: set rng outside before calling main function.
perm = randperm(n);
cv_id = zeros(n,1);
for i = 1:n
    cv_id(perm(i)) = mod(i-1, K) + 1;
end
end
