function S = predict_behavior_psd_elasticnet_globalAlpha(all_vects, all_behav, cov, C)
% Variant of your function using **Scheme B**:
%  1) Build full X (all subjects) once
%  2) Select a **global alpha** using CV on all subjects
%  3) Outer LOO: fix alpha, only tune lambda (inner CV on TRAIN)
%
% Notes:
% - For global-alpha selection, lasso does CV internally. MATLAB handles the
%   CV splits consistently if you pass a cvpartition object.
% - Outer loop still does train-only standardization (no leakage for the outer prediction).
%
% Requires: Statistics and Machine Learning Toolbox (lasso)

if nargin < 3, cov = []; end
if nargin < 4, C = struct(); end
if ~isfield(C,'ml'), C.ml = struct(); end

% -------------------- detect mode & sizes --------------------
nd = ndims(all_vects);
if nd == 2
    nROI  = size(all_vects,1);
    nSubj = size(all_vects,2);
    mode  = 'roi';
elseif nd == 3
    nROI  = size(all_vects,1);
    assert(size(all_vects,2) == nROI);
    nSubj = size(all_vects,3);
    mode  = 'edge';
else
    error('all_vects must be [nROI x nSubj] or [nROI x nROI x nSubj].');
end
assert(iscolumn(all_behav) && size(all_behav,1)==nSubj);

% -------------------- Elastic Net settings --------------------
alpha_grid = getfield_default(C.ml, 'enet_alpha_grid', [0.05 0.1 0.2 0.5 0.8 1.0]);
alpha_grid = alpha_grid(:)';

lambda_grid = getfield_default(C.ml, 'enet_lambda_grid', []);
if ~isempty(lambda_grid), lambda_grid = lambda_grid(:); end

Kfold = getfield_default(C.ml, 'enet_kfold', 5);

do_standardize = logical(getfield_default(C.ml, 'enet_standardize', true));
add_intercept  = logical(getfield_default(C.ml, 'enet_intercept', true));
use_cov        = logical(getfield_default(C.ml, 'enet_use_cov_as_features', false));
min_features   = getfield_default(C.ml, 'enet_min_features', 0);

% deterministic seeds (important)
seed_global = getfield_default(C.ml,'enet_seed_global_alpha', 12345);
seed_inner  = getfield_default(C.ml,'enet_seed_inner',       54321);

if use_cov
    assert(~isempty(cov) && size(cov,1)==nSubj);
    nCov = size(cov,2);
else
    nCov = 0;
end

% -------------------- feature mapping (ROI vs EDGE) --------------------
switch mode
    case 'roi'
        nBaseFeat = nROI;
        edge_idx = [];
        feat_mask_mat = [];
    case 'edge'
        edge_mask = getfield_default(C.ml, 'enet_edge_mask', []);
        use_triu  = logical(getfield_default(C.ml, 'enet_use_triu', true));
        if ~isempty(edge_mask)
            edge_mask = logical(edge_mask);
            assert(isequal(size(edge_mask), [nROI nROI]));
            feat_mask_mat = edge_mask;
        else
            feat_mask_mat = use_triu * triu(true(nROI),1) + (~use_triu) * true(nROI);
            feat_mask_mat = logical(feat_mask_mat);
        end
        edge_idx  = find(feat_mask_mat(:));
        nBaseFeat = numel(edge_idx);
end
nFeat = nBaseFeat + nCov;

% -------------------- Build FULL design matrix Xall (subjects x features) --------------------
Xall = nan(nSubj, nBaseFeat);

switch mode
    case 'roi'
        Xall = all_vects.'; % [nSubj x nROI]
    case 'edge'
        for s = 1:nSubj
            M = all_vects(:,:,s);
            v = M(:);
            Xall(s,:) = v(edge_idx).';
        end
end

if use_cov
    Xall = [Xall, cov];
end

yall = all_behav;

% -------------------- (B) Global alpha selection --------------------
% Use a fixed cvpartition for reproducibility
rng(seed_global);
if isinf(Kfold) || Kfold >= nSubj
    cvp_global = cvpartition(nSubj,'LeaveOut');
else
    nFoldsG = max(2, min(Kfold, nSubj));
    cvp_global = cvpartition(nSubj,'KFold', nFoldsG);
end

% Global selection criterion:
% - Use Index1SE (more stable) unless you explicitly set C.ml.enet_global_pick='min'
pick_rule = getfield_default(C.ml,'enet_global_pick','1se'); % '1se' or 'min'

best_alpha = NaN;
best_score = inf;

% IMPORTANT:
% For global-alpha selection, we let lasso standardize internally.
% (You can set this to false if you standardize Xall beforehand, but then
%  you’d be standardizing using all subjects.)
for a = alpha_grid
    if isempty(lambda_grid)
        [~, FitInfo] = lasso(Xall, yall, ...
            'Alpha', a, ...
            'CV', cvp_global, ...
            'Standardize', true, ...
            'Intercept', add_intercept);
    else
        [~, FitInfo] = lasso(Xall, yall, ...
            'Alpha', a, ...
            'Lambda', lambda_grid, ...
            'CV', cvp_global, ...
            'Standardize', true, ...
            'Intercept', add_intercept);
    end

    switch lower(pick_rule)
        case 'min'
            ii = FitInfo.IndexMinMSE;
        otherwise
            ii = FitInfo.Index1SE;
    end

    score = FitInfo.MSE(ii); % CV MSE at the chosen lambda for this alpha

    if score < best_score
        best_score = score;
        best_alpha = a;
    end
end

fprintf('Global alpha selected = %.4g (CV MSE=%.4g, rule=%s)\n', best_alpha, best_score, pick_rule);

% -------------------- Allocate outputs --------------------
pred_enet  = nan(nSubj,1);
alpha_sel  = nan(nSubj,1);
lambda_sel = nan(nSubj,1);

beta_all   = nan(nFeat, nSubj);
inter_all  = nan(nSubj,1);

mu_all = nan(nFeat, nSubj);
sd_all = nan(nFeat, nSubj);
nnz_all = nan(nSubj,1);

% -------------------- Outer LOO loop --------------------
parfor leftout = 1:nSubj
    train_idx = true(nSubj,1);
    train_idx(leftout) = false;

    Xtrain = Xall(train_idx, :);
    ytrain = yall(train_idx);
    Xtest  = Xall(leftout, :);

    nTrain = size(Xtrain,1);

    % ----- train-only standardization -----
    if do_standardize
        mu = mean(Xtrain, 1, 'omitnan');
        sd = std(Xtrain, 0, 1, 'omitnan');
        sd(sd==0 | isnan(sd)) = 1;
        XtrZ = (Xtrain - mu) ./ sd;
        XteZ = (Xtest  - mu) ./ sd;
    else
        mu = zeros(1,size(Xtrain,2));
        sd = ones(1,size(Xtrain,2));
        XtrZ = Xtrain;
        XteZ = Xtest;
    end

    % ----- inner CV folds on TRAIN only (deterministic) -----
    % Use a per-leftout seed so results are reproducible even in parfor
    rng(seed_inner + leftout);

    if isinf(Kfold) || Kfold >= nTrain
        cvp_inner = cvpartition(nTrain,'LeaveOut');
    else
        nFolds = max(2, min(Kfold, nTrain));
        cvp_inner = cvpartition(nTrain,'KFold', nFolds);
    end

    % ----- inner CV: alpha fixed, choose lambda -----
    a = best_alpha;

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

    % pick lambda: 1SE (stable) or min
    pick_inner = getfield_default(C.ml,'enet_inner_pick','1se'); % '1se'/'min'
    switch lower(pick_inner)
        case 'min'
            ii = FitInfo.IndexMinMSE;
        otherwise
            ii = FitInfo.Index1SE;
    end

    beta = B(:,ii);
    intercept = FitInfo.Intercept(ii);

    % min_features constraint (optional)
    if min_features > 0 && nnz(beta) < min_features
        % fallback to minMSE if 1SE too sparse
        ii2 = FitInfo.IndexMinMSE;
        beta2 = B(:,ii2);
        if nnz(beta2) >= min_features
            beta = beta2;
            intercept = FitInfo.Intercept(ii2);
            ii = ii2;
        end
    end

    % ----- save & predict -----
    alpha_sel(leftout)  = a;
    lambda_sel(leftout) = FitInfo.Lambda(ii);

    beta_all(:, leftout) = beta(:);
    inter_all(leftout)   = intercept;

    mu_all(:, leftout) = mu(:);
    sd_all(:, leftout) = sd(:);

    nnz_all(leftout) = nnz(beta);

    pred_enet(leftout) = XteZ * beta + intercept;
end

% -------------------- Evaluation --------------------
ok = ~isnan(pred_enet) & ~isnan(yall);
[r_enet, p_enet] = corr(pred_enet(ok), yall(ok));

se_enet  = (pred_enet - yall).^2;
mse_enet = mean(se_enet, 'omitnan');

cof_enet = regress(yall(ok), [pred_enet(ok), ones(sum(ok),1)]);

% -------------------- Pack outputs --------------------
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
S.meta.enet.kfold       = Kfold;
S.meta.enet.standardize = do_standardize;
S.meta.enet.intercept   = add_intercept;
S.meta.enet.use_cov_as_features = use_cov;
S.meta.enet.min_features = min_features;

S.meta.enet.scheme = 'B: global alpha via CV on all subjects; outer LOO fixes alpha, tunes lambda';
S.meta.enet.alpha_global = best_alpha;
S.meta.enet.global_pick_rule = pick_rule;

S.pred = struct();
S.pred.enet = pred_enet;

S.eval = struct();
S.eval.r_enet   = r_enet;
S.eval.p_enet   = p_enet;
S.eval.mse_enet = mse_enet;
S.eval.se_enet  = se_enet;
S.eval.calib_enet = cof_enet;

S.model = struct();
S.model.alpha_global      = best_alpha;
S.model.alpha_perfold     = alpha_sel;      % will be constant = best_alpha
S.model.lambda_perfold    = lambda_sel;
S.model.beta_perfold      = beta_all;
S.model.intercept_perfold = inter_all;

S.debug = struct();
S.debug.mu_perfold = mu_all;
S.debug.sd_perfold = sd_all;
S.debug.nnz_perfold = nnz_all;

if strcmp(mode,'edge')
    S.model.edge_idx = edge_idx;
    S.model.feat_mask_mat = feat_mask_mat;
end

end

% ====================== helpers ======================
function val = getfield_default(S, fname, default_val)
if isstruct(S) && isfield(S, fname) && ~isempty(S.(fname))
    val = S.(fname);
else
    val = default_val;
end
end
