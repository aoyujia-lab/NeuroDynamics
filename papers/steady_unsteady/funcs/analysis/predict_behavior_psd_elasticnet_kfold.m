function S = predict_behavior_psd_elasticnet_kfold(all_vects, all_behav, cov, C)
% K-fold (repeated) nested CV Elastic Net for ROI or EDGE features.
%
% INPUT
%   all_vects : [nROI x nSubj] OR [nROI x nROI x nSubj]
%   all_behav : [nSubj x 1]
%   cov       : [nSubj x nCov] (optional; used if C.ml.enet_use_cov_as_features=true)
%   C         : config struct
%
% OUTPUT (struct S)
%   S.meta.*, S.rep(r).pred.*, S.rep(r).eval.*, S.rep(r).model.*, S.summary.*
%
% Notes
% - Outer CV: K-fold (repeated)
% - Inner CV: K-fold (within TRAIN only) to select alpha+lambda
% - Standardization: TRAIN-only
% - Parallel: outerRepeat can be parallelized via parfor
%
% Requires: Statistics and Machine Learning Toolbox (lasso)
%           Parallel Computing Toolbox (if use_parallel=true)

% -------------------- defaults --------------------
if nargin < 3, cov = []; end
if nargin < 4, C = struct(); end
if ~isfield(C,'ml'), C.ml = struct(); end

% -------------------- detect mode & sizes --------------------
nd = ndims(all_vects);
if nd == 2
    assert(ismatrix(all_vects), 'all_vects must be 2D [nROI x nSubj] or 3D [nROI x nROI x nSubj].');
    nROI  = size(all_vects,1);
    nSubj = size(all_vects,2);
    mode  = 'roi';
elseif nd == 3
    nROI  = size(all_vects,1);
    assert(size(all_vects,2) == nROI, '3D all_vects must be [nROI x nROI x nSubj].');
    nSubj = size(all_vects,3);
    mode  = 'edge';
else
    error('all_vects must be [nROI x nSubj] or [nROI x nROI x nSubj].');
end
assert(iscolumn(all_behav) && size(all_behav,1) == nSubj, 'all_behav must be [nSubj x 1].');

% -------------------- Elastic Net settings --------------------
alpha_grid = getfield_default(C.ml, 'enet_alpha_grid', [0 0.05 0.1 0.2 0.5 0.8 1.0]);
alpha_grid = alpha_grid(:)';

lambda_grid = getfield_default(C.ml, 'enet_lambda_grid', []);
if ~isempty(lambda_grid), lambda_grid = lambda_grid(:); end

innerK = getfield_default(C.ml, 'enet_kfold', 3);  % inner folds
do_standardize = logical(getfield_default(C.ml, 'enet_standardize', true));
add_intercept  = logical(getfield_default(C.ml, 'enet_intercept', true));
use_cov        = logical(getfield_default(C.ml, 'enet_use_cov_as_features', false));
min_features   = getfield_default(C.ml, 'enet_min_features', 0);
pick_rule      = lower(getfield_default(C.ml, 'enet_pick_rule', 'minmse')); % 'minmse' or '1se'

% -------------------- Outer CV settings --------------------
outerK      = getfield_default(C.ml, 'outer_kfold', 5);
outerRepeat = getfield_default(C.ml, 'outer_repeat', 50);
outerSeed   = getfield_default(C.ml, 'outer_seed', 1);
outerStratify = logical(getfield_default(C.ml, 'outer_stratify', false)); % see note below

% optional: control inner fold RNG too (only matters because we explicitly use cvpartition)
innerSeed = getfield_default(C.ml, 'inner_seed', []); % [] -> do not reset rng for inner

% -------------------- Parallel settings --------------------
use_parallel = logical(getfield_default(C.ml, 'use_parallel', true));
nWorkers     = getfield_default(C.ml, 'nWorkers', []);  % [] -> default

if use_parallel
    pool = gcp('nocreate');
    if isempty(pool)
        if isempty(nWorkers)
            parpool;
        else
            parpool(nWorkers);
        end
    end
end

% cov handling
if use_cov
    assert(~isempty(cov) && size(cov,1) == nSubj, ...
        'cov must be [nSubj x nCov] when enet_use_cov_as_features=true.');
    nCov = size(cov,2);
else
    nCov = 0;
end

% -------------------- feature mapping (ROI vs EDGE) --------------------
switch mode
    case 'roi'
        nBaseFeat = nROI;
        feat_mask_mat = [];
        edge_idx = [];
        roi_idx  = (1:nROI)';

    case 'edge'
        edge_mask = getfield_default(C.ml, 'enet_edge_mask', []);
        use_triu  = logical(getfield_default(C.ml, 'enet_use_triu', true));

        if ~isempty(edge_mask)
            edge_mask = logical(edge_mask);
            assert(isequal(size(edge_mask), [nROI nROI]), 'C.ml.enet_edge_mask must be [nROI x nROI].');
            feat_mask_mat = edge_mask;
        else
            if use_triu
                feat_mask_mat = triu(true(nROI), 1);
            else
                feat_mask_mat = true(nROI);
            end
        end
        edge_idx  = find(feat_mask_mat(:));
        nBaseFeat = numel(edge_idx);
        roi_idx   = [];
end

nFeat = nBaseFeat + nCov;

% -------------------- Meta --------------------
S = struct();
S.meta = struct();
S.meta.mode  = mode;
S.meta.nROI  = nROI;
S.meta.nSubj = nSubj;
S.meta.nBaseFeat = nBaseFeat;
S.meta.nCov  = nCov;
S.meta.nFeat = nFeat;

S.meta.enet = struct();
S.meta.enet.alpha_grid  = alpha_grid;
S.meta.enet.lambda_grid = lambda_grid;
S.meta.enet.inner_kfold = innerK;
S.meta.enet.standardize = do_standardize;
S.meta.enet.intercept   = add_intercept;
S.meta.enet.use_cov_as_features = use_cov;
S.meta.enet.min_features = min_features;
S.meta.enet.pick_rule = pick_rule;

S.meta.outer = struct();
S.meta.outer.kfold   = outerK;
S.meta.outer.repeat  = outerRepeat;
S.meta.outer.seed    = outerSeed;
S.meta.outer.stratify = outerStratify;

S.meta.parallel = struct();
S.meta.parallel.use_parallel = use_parallel;
S.meta.parallel.nWorkers = ternary(isempty(nWorkers), NaN, nWorkers);

if strcmp(mode,'edge')
    S.meta.enet.use_triu = logical(getfield_default(C.ml, 'enet_use_triu', true));
    S.meta.enet.edge_mask_provided = ~isempty(getfield_default(C.ml, 'enet_edge_mask', []));
end

fprintf('Elastic Net (%s mode): Outer %d-fold x %d repeats, n=%d\n', upper(mode), outerK, outerRepeat, nSubj);
fprintf('  base features: %d%s, cov: %d, total: %d\n', nBaseFeat, ternary(strcmp(mode,'edge'),' edges',' ROI'), nCov, nFeat);
fprintf('  inner CV folds: %d, pick_rule: %s\n', min(innerK, nSubj-1), pick_rule);
fprintf('  parallel: %d\n', use_parallel);

% -------------------- Run repeated outer K-fold --------------------
% parfor-safe containers
rep_cell = cell(outerRepeat,1);
r_all = nan(outerRepeat,1);
p_all = nan(outerRepeat,1);
mse_all = nan(outerRepeat,1);

baseSeed = outerSeed;

if use_parallel
    parfor rr = 1:outerRepeat
        % Make each repeat reproducible + independent
        rng(baseSeed + rr, 'twister');

        % Outer partition
        if outerStratify
            % NOTE: for continuous y, this is not truly stratified in the usual sense.
            % If you want proper stratification, discretize y outside and pass that label.
            cvp_outer = cvpartition(all_behav, 'KFold', outerK);
        else
            cvp_outer = cvpartition(nSubj, 'KFold', outerK);
        end

        [Srep, r_enet, p_enet, mse] = run_one_repeat( ...
            cvp_outer, all_vects, all_behav, cov, C, mode, ...
            nROI, nSubj, edge_idx, nBaseFeat, nFeat, use_cov, ...
            do_standardize, add_intercept, alpha_grid, lambda_grid, ...
            innerK, min_features, pick_rule, innerSeed);

        rep_cell{rr} = Srep;
        r_all(rr) = r_enet;
        p_all(rr) = p_enet;
        mse_all(rr) = mse;
    end
else
    for rr = 1:outerRepeat
        rng(baseSeed + rr, 'twister');

        if outerStratify
            cvp_outer = cvpartition(all_behav, 'KFold', outerK);
        else
            cvp_outer = cvpartition(nSubj, 'KFold', outerK);
        end

        [Srep, r_enet, p_enet, mse] = run_one_repeat( ...
            cvp_outer, all_vects, all_behav, cov, C, mode, ...
            nROI, nSubj, edge_idx, nBaseFeat, nFeat, use_cov, ...
            do_standardize, add_intercept, alpha_grid, lambda_grid, ...
            innerK, min_features, pick_rule, innerSeed);

        rep_cell{rr} = Srep;
        r_all(rr) = r_enet;
        p_all(rr) = p_enet;
        mse_all(rr) = mse;
    end
end

% assign back to S
S.rep = vertcat(rep_cell{:});

% -------------------- Summary across repeats --------------------
S.summary = struct();
S.summary.r_mean   = mean(r_all, 'omitnan');
S.summary.r_std    = std(r_all,  'omitnan');
S.summary.r_median = median(r_all, 'omitnan');
S.summary.p_mean   = mean(p_all, 'omitnan');
S.summary.mse_mean = mean(mse_all, 'omitnan');
S.summary.mse_std  = std(mse_all, 'omitnan');

S.summary.r_all = r_all;
S.summary.p_all = p_all;
S.summary.mse_all = mse_all;

% mapping info
S.model = struct();
S.model.mode = mode;
if strcmp(mode,'roi')
    S.model.roi_idx = roi_idx;
else
    S.model.edge_idx = edge_idx;
    S.model.feat_mask_mat = feat_mask_mat;
end

end

% ====================== repeat runner ======================
function [Srep, r_enet, p_enet, mse] = run_one_repeat( ...
    cvp_outer, all_vects, all_behav, cov, C, mode, ...
    nROI, nSubj, edge_idx, nBaseFeat, nFeat, use_cov, ...
    do_standardize, add_intercept, alpha_grid, lambda_grid, ...
    innerK, min_features, pick_rule, innerSeed)

pred = nan(nSubj,1);

alpha_sel  = nan(nSubj,1);
lambda_sel = nan(nSubj,1);

beta_perSub  = nan(nFeat, nSubj);
inter_perSub = nan(nSubj,1);

mu_perSub = nan(nFeat, nSubj);
sd_perSub = nan(nFeat, nSubj);
nnz_perSub = nan(nSubj,1);

for fold = 1:cvp_outer.NumTestSets
    test_idx  = test(cvp_outer, fold);
    train_idx = training(cvp_outer, fold);

    out_fold = fit_predict_one_fold(all_vects, all_behav, cov, C, mode, ...
        train_idx, test_idx, nROI, edge_idx, nBaseFeat, ...
        use_cov, do_standardize, add_intercept, ...
        alpha_grid, lambda_grid, innerK, min_features, pick_rule, innerSeed);

    pred(test_idx) = out_fold.pred;

    % same model applied to all test subjects in this fold
    alpha_sel(test_idx)  = out_fold.alpha;
    lambda_sel(test_idx) = out_fold.lambda;

    beta_perSub(:, test_idx)  = repmat(out_fold.beta(:), 1, sum(test_idx));
    inter_perSub(test_idx)    = out_fold.intercept;

    mu_perSub(:, test_idx) = repmat(out_fold.mu(:), 1, sum(test_idx));
    sd_perSub(:, test_idx) = repmat(out_fold.sd(:), 1, sum(test_idx));
    nnz_perSub(test_idx)   = out_fold.nnz;
end

[r_enet, p_enet] = corr(pred, all_behav, 'rows', 'pairwise');
se  = (pred - all_behav).^2;
mse = mean(se, 'omitnan');

Srep = struct();
Srep.pred  = struct('enet', pred);
Srep.eval  = struct('r_enet', r_enet, 'p_enet', p_enet, 'mse_enet', mse);
Srep.model = struct();
Srep.model.alpha_perSub  = alpha_sel;
Srep.model.lambda_perSub = lambda_sel;
Srep.model.beta_perSub   = beta_perSub;
Srep.model.intercept_perSub = inter_perSub;
Srep.debug = struct();
Srep.debug.mu_perSub = mu_perSub;
Srep.debug.sd_perSub = sd_perSub;
Srep.debug.nnz_perSub = nnz_perSub;

end

% ====================== fold fit/predict ======================
function out = fit_predict_one_fold(all_vects, all_behav, cov, C, mode, ...
    train_idx, test_idx, nROI, edge_idx, nBaseFeat, ...
    use_cov, do_standardize, add_intercept, ...
    alpha_grid, lambda_grid, innerK, min_features, pick_rule, innerSeed)

ytrain = all_behav(train_idx);

% ----- build Xtrain/Xtest depending on mode -----
switch mode
    case 'roi'
        Xtrain = all_vects(:, train_idx)';      % [nTrain x nROI]
        Xtest  = all_vects(:, test_idx)';       % [nTest x nROI]

    case 'edge'
        tr_list = find(train_idx);
        te_list = find(test_idx);

        nTrain = numel(tr_list);
        nTest  = numel(te_list);

        Xtrain = nan(nTrain, nBaseFeat);
        for t = 1:nTrain
            M = all_vects(:,:, tr_list(t));
            v = M(:);
            Xtrain(t,:) = v(edge_idx)';
        end

        Xtest = nan(nTest, nBaseFeat);
        for t = 1:nTest
            M = all_vects(:,:, te_list(t));
            v = M(:);
            Xtest(t,:) = v(edge_idx)';
        end
end

if use_cov
    Xtrain = [Xtrain, cov(train_idx,:)];
    Xtest  = [Xtest,  cov(test_idx,:)];
end

% ----- standardize using TRAIN only -----
if do_standardize
    mu = mean(Xtrain, 1, 'omitnan');
    sd = std(Xtrain, 0, 1, 'omitnan');
    sd(sd == 0 | isnan(sd)) = 1;
    XtrZ = (Xtrain - mu) ./ sd;
    XteZ = (Xtest  - mu) ./ sd;
else
    mu = zeros(1,size(Xtrain,2));
    sd = ones(1,size(Xtrain,2));
    XtrZ = Xtrain;
    XteZ = Xtest;
end

% ----- inner CV folds (on TRAIN only) -----
nTrain = sum(train_idx);
nFolds = max(2, min(innerK, nTrain));

if ~isempty(innerSeed)
    rng(innerSeed, 'twister');
end
cvp_inner = cvpartition(nTrain, 'KFold', nFolds);

% ----- inner CV: choose alpha + lambda -----
best = struct('mse', inf, 'alpha', NaN, 'lambda', NaN, 'beta', [], 'intercept', NaN, 'nnz', NaN);

for a = alpha_grid
    if isempty(lambda_grid)
        [B, FitInfo] = lasso(XtrZ, ytrain, ...
            'Alpha', a, ...
            'CV', cvp_inner, ...
            'Standardize', false, ...
            'Intercept', add_intercept);
    else
        [B, FitInfo] = lasso(XtrZ, ytrain, ...
            'Alpha', a, ...
            'Lambda', lambda_grid, ...
            'CV', cvp_inner, ...
            'Standardize', false, ...
            'Intercept', add_intercept);
    end

    switch pick_rule
        case '1se'
            idx = FitInfo.Index1SE;
        otherwise
            idx = FitInfo.IndexMinMSE;
    end

    beta = B(:, idx);
    intercept = FitInfo.Intercept(idx);
    cur_nnz = nnz(beta);
    cur_mse = FitInfo.MSE(idx);

    if min_features > 0 && cur_nnz < min_features
        continue;
    end

    if cur_mse < best.mse
        best.mse = cur_mse;
        best.alpha = a;
        best.lambda = FitInfo.Lambda(idx);
        best.beta = beta;
        best.intercept = intercept;
        best.nnz = cur_nnz;
    end
end

% fallback if min_features filtered everything
if isempty(best.beta)
    best = struct('mse', inf, 'alpha', NaN, 'lambda', NaN, 'beta', [], 'intercept', NaN, 'nnz', NaN);
    for a = alpha_grid
        if isempty(lambda_grid)
            [B, FitInfo] = lasso(XtrZ, ytrain, 'Alpha', a, 'CV', cvp_inner, 'Standardize', false, 'Intercept', add_intercept);
        else
            [B, FitInfo] = lasso(XtrZ, ytrain, 'Alpha', a, 'Lambda', lambda_grid, 'CV', cvp_inner, 'Standardize', false, 'Intercept', add_intercept);
        end
        idx = FitInfo.IndexMinMSE;
        cur_mse = FitInfo.MSE(idx);
        if cur_mse < best.mse
            best.mse = cur_mse;
            best.alpha = a;
            best.lambda = FitInfo.Lambda(idx);
            best.beta = B(:, idx);
            best.intercept = FitInfo.Intercept(idx);
            best.nnz = nnz(best.beta);
        end
    end
end

% ----- predict test subjects -----
pred = XteZ * best.beta + best.intercept;

out = struct();
out.pred = pred;
out.alpha = best.alpha;
out.lambda = best.lambda;
out.beta = best.beta(:);
out.intercept = best.intercept;
out.mu = mu(:);
out.sd = sd(:);
out.nnz = best.nnz;

end

% ====================== misc helpers ======================
function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end

function val = getfield_default(S, fname, default_val)
if isstruct(S) && isfield(S, fname) && ~isempty(S.(fname))
    val = S.(fname);
else
    val = default_val;
end
end
