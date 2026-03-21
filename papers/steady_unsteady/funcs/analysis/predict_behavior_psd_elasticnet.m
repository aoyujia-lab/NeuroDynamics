function S = predict_behavior_psd_elasticnet(all_vects, all_behav, cov, C)
% LOO elastic net with inner repeated CV for alpha/lambda selection.
% all_vects: [nROI x nSubj] OR [nROI x nROI x nSubj]
% all_behav: [nSubj x 1]
% cov      : [nSubj x nCov] optional
% C.ml.*   : see defaults below

if nargin < 3, cov = []; end
if nargin < 4, C = struct(); end
if ~isfield(C,'ml'), C.ml = struct(); end

% --------- parse mode & sizes ----------
[mode, nROI, nSubj] = detect_mode(all_vects);
assert(iscolumn(all_behav) && numel(all_behav)==nSubj, 'all_behav must be [nSubj x 1].');

% --------- config defaults ----------
ml = C.ml;
alpha_grid = def(ml,'enet_alpha_grid',[0.05 0.1 0.2 0.5 0.8 1.0]); alpha_grid = alpha_grid(:)';
lambda_grid = def(ml,'enet_lambda_grid',[]); if ~isempty(lambda_grid), lambda_grid=lambda_grid(:); end
Kfold   = def(ml,'enet_kfold',5);
nRepeat = def(ml,'enet_repeat',10);
seed0   = def(ml,'enet_seed',1);
use_1se = logical(def(ml,'enet_use_1se',false));
do_z    = logical(def(ml,'enet_standardize',true));
add_int = logical(def(ml,'enet_intercept',true));
use_cov = logical(def(ml,'enet_use_cov_as_features',false));
min_nnz = def(ml,'enet_min_features',0);


% edge mask / indexing
edge_idx = [];
feat_mask_mat = [];
if strcmp(mode,'edge')
    [edge_idx, feat_mask_mat] = edge_feature_index(nROI, ml);
end

% cov
if use_cov
    assert(~isempty(cov) && size(cov,1)==nSubj, 'cov must be [nSubj x nCov].');
    nCov = size(cov,2);
else
    nCov = 0;
end

nBaseFeat = strcmp(mode,'roi') * nROI + strcmp(mode,'edge') * numel(edge_idx);
nFeat = nBaseFeat + nCov;

% --------- allocate ----------
pred = nan(nSubj,1);
alpha_sel = nan(nSubj,1);
lambda_sel = nan(nSubj,1);
beta_all = nan(nFeat, nSubj);
inter_all = nan(nSubj,1);
mu_all = nan(nFeat, nSubj);
sd_all = nan(nFeat, nSubj);
nnz_all = nan(nSubj,1);

% --------- outer LOO ----------
parfor leftout = 1:nSubj
    train_idx = true(nSubj,1); train_idx(leftout)=false;
    ytr = all_behav(train_idx);
    nTrain = sum(train_idx);

    % build features
    Xtr = build_X(all_vects, train_idx, mode, edge_idx);  % [nTrain x nBaseFeat]
    Xte = build_X(all_vects, leftout,  mode, edge_idx);   % [1 x nBaseFeat]
    if use_cov
        Xtr = [Xtr, cov(train_idx,:)];
        Xte = [Xte, cov(leftout,:)];
    end

    % standardize train-only
    [XtrZ, XteZ, mu, sd] = standardize_train_only(Xtr, Xte, do_z);

    % choose inner folds
    nFolds = pick_nfolds(Kfold, nTrain);

    % inner repeated CV select alpha/lambda
    best = select_enet_repeated_cv(XtrZ, ytr, alpha_grid, lambda_grid, nFolds, ...
        nRepeat, seed0, leftout, add_int, use_1se, min_nnz);

    % predict
    pred(leftout) = XteZ * best.beta + best.intercept;

    % store
    alpha_sel(leftout)  = best.alpha;
    lambda_sel(leftout) = best.lambda;
    beta_all(:,leftout) = best.beta(:);
    inter_all(leftout)  = best.intercept;
    mu_all(:,leftout)   = mu(:);
    sd_all(:,leftout)   = sd(:);
    nnz_all(leftout)    = best.nnz;
end

% --------- eval ----------
[r_enet, p_enet] = corr(pred, all_behav, 'rows','pairwise');

err = pred - all_behav;
se  = err.^2;
mse = mean(se,'omitnan');

% CV-R^2 (Q^2): 1 - SSE/SST, consistent with MSE version
ok  = ~isnan(pred) & ~isnan(all_behav);
y   = all_behav(ok);
yh  = pred(ok);

SSE = sum((y - yh).^2);
SST = sum((y - mean(y)).^2);

if SST > 0
    Q2 = 1 - SSE / SST;
else
    Q2 = NaN;  % degenerate case: y has zero variance
end

calib = regress(all_behav, [pred, ones(nSubj,1)]);

% --------- pack outputs ----------
S = struct();
S.meta = struct('mode',mode,'nROI',nROI,'nSubj',nSubj,'nBaseFeat',nBaseFeat,'nCov',nCov,'nFeat',nFeat);
S.meta.enet = struct('alpha_grid',alpha_grid,'lambda_grid',lambda_grid,'kfold',Kfold,'repeat',nRepeat, ...
    'standardize',do_z,'intercept',add_int,'use_cov_as_features',use_cov,'min_features',min_nnz,'use_1se',use_1se);

if strcmp(mode,'edge')
    S.meta.enet.use_triu = logical(def(ml,'enet_use_triu',true));
    S.meta.enet.edge_mask_provided = ~isempty(def(ml,'enet_edge_mask',[]));
end

S.pred = struct('enet',pred);
S.eval = struct('r_enet',r_enet,'p_enet',p_enet, ...
                'mse_enet',mse,'se_enet',se, ...
                'Q2_enet',Q2, ...
                'calib_enet',calib);
S.model = struct();
S.model.mode = mode;
S.model.alpha_perfold  = alpha_sel;
S.model.lambda_perfold = lambda_sel;
S.model.beta_perfold   = beta_all;
S.model.intercept_perfold = inter_all;

if strcmp(mode,'roi')
    S.model.roi_idx = (1:nROI)'; % feature i -> ROI index
else
    S.model.edge_idx = edge_idx;
    S.model.feat_mask_mat = feat_mask_mat;
end

S.debug = struct('mu_perfold',mu_all,'sd_perfold',sd_all,'nnz_perfold',nnz_all);

end

% ======================= helpers =======================

function [mode, nROI, nSubj] = detect_mode(all_vects)
nd = ndims(all_vects);
if nd==2
    nROI = size(all_vects,1); nSubj=size(all_vects,2); mode='roi';
elseif nd==3
    nROI=size(all_vects,1);
    assert(size(all_vects,2)==nROI,'3D all_vects must be [nROI x nROI x nSubj].');
    nSubj=size(all_vects,3); mode='edge';
else
    error('all_vects must be 2D or 3D.');
end
end

function [edge_idx, feat_mask_mat] = edge_feature_index(nROI, ml)
edge_mask = def(ml,'enet_edge_mask',[]);
use_triu  = logical(def(ml,'enet_use_triu',true));
if ~isempty(edge_mask)
    edge_mask = logical(edge_mask);
    assert(isequal(size(edge_mask),[nROI nROI]), 'enet_edge_mask must be [nROI x nROI].');
    feat_mask_mat = edge_mask;
else
    feat_mask_mat = use_triu * triu(true(nROI),1) + (~use_triu) * true(nROI);
    feat_mask_mat = logical(feat_mask_mat);
end
edge_idx = find(feat_mask_mat(:));
end

function X = build_X(all_vects, idx, mode, edge_idx)
if strcmp(mode,'roi')
    if islogical(idx)
        X = all_vects(:,idx)';     % [nSub x nROI]
    else
        X = all_vects(:,idx)';     % [1 x nROI]
    end
else
    if islogical(idx)
        subs = find(idx);
    else
        subs = idx;
    end
    nS = numel(subs);
    X = nan(nS, numel(edge_idx));
    for i = 1:nS
        v = all_vects(:,:,subs(i)); v = v(:);
        X(i,:) = v(edge_idx)';
    end
end
end

function [XtrZ, XteZ, mu, sd] = standardize_train_only(Xtr, Xte, do_z)
if do_z
    mu = mean(Xtr,1,'omitnan');
    sd = std(Xtr,0,1,'omitnan');
    sd(sd==0 | isnan(sd)) = 1;
    XtrZ = (Xtr - mu) ./ sd;
    XteZ = (Xte - mu) ./ sd;
else
    mu = zeros(1,size(Xtr,2));
    sd = ones(1,size(Xtr,2));
    XtrZ = Xtr; XteZ = Xte;
end
end

function nFolds = pick_nfolds(Kfold, nTrain)
if isinf(Kfold) || Kfold >= nTrain
    nFolds = nTrain; % LOO
else
    nFolds = max(2, min(Kfold, nTrain));
end
end

function best = select_enet_repeated_cv(XtrZ, ytr, alpha_grid, lambda_grid, nFolds, ...
    nRepeat, seed0, leftout, add_int, use_1se, min_nnz)

% collect proposals
prop_a = []; prop_l = []; prop_mse = [];
prop_beta = {}; prop_int = []; prop_nnz = [];

for rr = 1:nRepeat
    rng(seed0 + 1000*leftout + rr, 'twister');
    n = numel(ytr);
if nFolds >= n
    cvp = cvpartition(n,'KFold',n);      % LOO, but lasso-compatible
else
    cvp = cvpartition(n,'KFold',nFolds);
end

    for a = alpha_grid
        args = {'Alpha',a,'CV',cvp,'Standardize',false,'Intercept',add_int};
        if ~isempty(lambda_grid), args = [args, {'Lambda',lambda_grid}]; end
        [B, FitInfo] = lasso(XtrZ, ytr, args{:});

        ii = FitInfo.IndexMinMSE;
        if use_1se, ii = FitInfo.Index1SE; end

        beta = B(:,ii);
        nnz_beta = nnz(beta);
        if min_nnz > 0 && nnz_beta < min_nnz
            continue
        end

        prop_a(end+1,1)   = a;
        prop_l(end+1,1)   = FitInfo.Lambda(ii);
        prop_mse(end+1,1) = FitInfo.MSE(ii);
        prop_beta{end+1,1} = beta;
        prop_int(end+1,1) = FitInfo.Intercept(ii);
        prop_nnz(end+1,1) = nnz_beta;
    end
end

% fallback: if min_nnz was too strict, rerun once with min_nnz=0
if isempty(prop_mse) && min_nnz > 0
    best = select_enet_repeated_cv(XtrZ, ytr, alpha_grid, lambda_grid, nFolds, ...
        nRepeat, seed0, leftout, add_int, use_1se, 0);
    return
end

% aggregate by unique (alpha, lambda) pair; choose smallest mean MSE
pair = [prop_a, prop_l];
[pair_u,~,ic] = unique(pair,'rows','stable');
mean_mse = accumarray(ic, prop_mse, [], @mean);

[~, iu] = min(mean_mse);
best_pair = pair_u(iu,:);

% pick one snapshot among matching proposals (smallest single-run MSE)
match = (prop_a==best_pair(1)) & (prop_l==best_pair(2));
idxs = find(match);
[~, k] = min(prop_mse(match));
pick = idxs(k);

best = struct();
best.alpha     = prop_a(pick);
best.lambda    = prop_l(pick);
best.mse       = mean_mse(iu);
best.beta      = prop_beta{pick};
best.intercept = prop_int(pick);
best.nnz       = prop_nnz(pick);
end

function v = def(S,f,default_v)
if isstruct(S) && isfield(S,f) && ~isempty(S.(f))
    v = S.(f);
else
    v = default_v;
end
end
