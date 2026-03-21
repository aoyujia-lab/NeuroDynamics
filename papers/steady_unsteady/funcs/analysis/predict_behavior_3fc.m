function S = predict_behavior_fc(all_fc, roi_feat1, roi_feat2, all_behav, cov, C)
% Multi-feature CPM (FC edges + arbitrary feature blocks) with LOO CV.
%
% INPUT
%   all_fc    : [nROI x nROI x nSubj]
%   roi_feat1 : [nFeat1 x nSubj]   (arbitrary feature block)
%   roi_feat2 : [nFeat2 x nSubj]   (arbitrary feature block)
%   all_behav : [nSubj x 1]
%   cov       : [nSubj x nCov] (required when fs_option = 4)
%   C         : config
%
% OUTPUT: struct S

% -------------------- Input --------------------
fs_option = C.ml.corr_option;
thresh    = C.ml.alpha;
Kfallback = C.ml.Kfallback;
use_fallback = (Kfallback > 0);

if isfield(C.ml,'zscore_each_fold')
    zfold = logical(C.ml.zscore_each_fold);
else
    zfold = true;
end

assert(ndims(all_fc) == 3, 'all_fc must be [nROI x nROI x nSubj].');
[nROI, nROI2, nSubj] = size(all_fc);
assert(nROI == nROI2, 'all_fc first two dims must be square.');
assert(size(all_behav,1) == nSubj, 'all_behav must be [nSubj x 1].');

assert(size(roi_feat1,2) == nSubj, 'roi_feat1 must be [nFeat1 x nSubj].');
assert(size(roi_feat2,2) == nSubj, 'roi_feat2 must be [nFeat2 x nSubj].');

if fs_option == 4
    assert(~isempty(cov) && size(cov,1) == nSubj, ...
        'cov must be [nSubj x nCov] when fs_option=4.');
end

% -------------------- FC vectorization --------------------
utMask   = triu(true(nROI), 1);
feat_idx = find(utMask);
nFeat_fc = numel(feat_idx);

X_fc = zeros(nFeat_fc, nSubj);
for s = 1:nSubj
    mat = all_fc(:,:,s);
    X_fc(:,s) = double(mat(feat_idx));
end

% -------------------- arbitrary feature blocks --------------------
X_feat1 = double(roi_feat1);   % [nFeat1 x nSubj]
X_feat2 = double(roi_feat2);   % [nFeat2 x nSubj]

nFeat1 = size(X_feat1,1);
nFeat2 = size(X_feat2,1);

% -------------------- concatenate all features --------------------
Xfeat = [X_fc; X_feat1; X_feat2];   % [nFeat_total x nSubj]
nFeat = size(Xfeat,1);

% -------------------- Meta --------------------
S = struct();
S.meta = struct();
S.meta.nROI   = nROI;
S.meta.nSubj  = nSubj;
S.meta.nFeat  = nFeat;
S.meta.nFeat_fc   = nFeat_fc;
S.meta.nFeat_feat1 = nFeat1;
S.meta.nFeat_feat2 = nFeat2;
S.meta.thresh = thresh;
S.meta.fs_option = fs_option;
S.meta.Kfallback = Kfallback;
S.meta.zscore_each_fold = zfold;

S.meta.featureBlocks = struct( ...
    'fc',    1:nFeat_fc, ...
    'feat1', nFeat_fc + (1:nFeat1), ...
    'feat2', nFeat_fc + nFeat1 + (1:nFeat2) ...
);

fprintf('Multi-feature CPM: FC(%d) + feat1(%d) + feat2(%d) = total %d features\n', ...
    nFeat_fc, nFeat1, nFeat2, nFeat);

% -------------------- Allocate --------------------
pred_pos = zeros(nSubj,1);
pred_neg = zeros(nSubj,1);
pred_com = zeros(nSubj,1);

b_pos_all = nan(nSubj,2);
b_neg_all = nan(nSubj,2);
b_com_all = nan(nSubj,3);

pos_masks = false(nFeat, nSubj);
neg_masks = false(nFeat, nSubj);
com_masks = false(nFeat, nSubj);

n_pos = zeros(nSubj,1);
n_neg = zeros(nSubj,1);
n_com = zeros(nSubj,1);

% -------------------- LOO loop --------------------
for leftout = 1:nSubj
    train_idx = true(nSubj,1);
    train_idx(leftout) = false;

    Xtrain = Xfeat(:, train_idx);
    ytrain = all_behav(train_idx);
    nTrain = sum(train_idx);

    % ---- zscore per fold ----
    if zfold
        mu = mean(Xtrain, 2, 'omitnan');
        sd = std(Xtrain, 0, 2, 'omitnan');
        sd(sd==0 | isnan(sd)) = 1;
        Xtrain_z = (Xtrain - mu) ./ sd;
        Xtest_z  = (Xfeat(:,leftout) - mu) ./ sd;
    else
        Xtrain_z = Xtrain;
        Xtest_z  = Xfeat(:,leftout);
    end

    % ---- feature selection ----
    switch fs_option
        case 1
            [r_vec, p_vec] = corr(Xtrain_z', ytrain);
        case 2
            [r_vec, p_vec] = corr(Xtrain_z', ytrain, 'type', 'Spearman');
        case 3
            warning('off');
            r_vec = zeros(nFeat,1);
            p_vec = zeros(nFeat,1);
            for i = 1:nFeat
                [~, stats] = robustfit(Xtrain_z(i,:)', ytrain);
                cur_t = stats.t(2);
                df = nTrain - 2;
                r_vec(i) = sign(cur_t) * sqrt(cur_t^2 / (df + cur_t^2));
                p_vec(i) = 2 * (1 - tcdf(abs(cur_t), df));
            end
            warning('on');
        case 4
            cov_train = cov(train_idx,:);
            [r_vec, p_vec] = partialcorr(Xtrain_z', ytrain, cov_train);
        otherwise
            error('Unknown fs_option.');
    end

    % ---- masks + fallback ----
    pos_sig = (r_vec > 0) & (p_vec < thresh);
    neg_sig = (r_vec < 0) & (p_vec < thresh);
    com_sig = (p_vec < thresh);

    if use_fallback && nnz(pos_sig) < Kfallback
        pos_mask = topk_mask_by_p(p_vec, r_vec > 0, Kfallback);
    else
        pos_mask = pos_sig;
    end

    if use_fallback && nnz(neg_sig) < Kfallback
        neg_mask = topk_mask_by_p(p_vec, r_vec < 0, Kfallback);
    else
        neg_mask = neg_sig;
    end

    if use_fallback && nnz(com_sig) < Kfallback
        com_mask = topk_mask_by_p(p_vec, true(nFeat,1), Kfallback);
    else
        com_mask = com_sig;
    end

    pos_masks(:,leftout) = pos_mask;
    neg_masks(:,leftout) = neg_mask;
    com_masks(:,leftout) = com_mask;

    n_pos(leftout) = nnz(pos_mask);
    n_neg(leftout) = nnz(neg_mask);
    n_com(leftout) = nnz(com_mask);

    % ---- sum features ----
    sumpos_train = sum(Xtrain_z(pos_mask,:),1)';
    sumneg_train = sum(Xtrain_z(neg_mask,:),1)';

    sumpos_test = sum(Xtest_z(pos_mask));
    sumneg_test = sum(Xtest_z(neg_mask));

    com_pos_mask = com_mask & (r_vec > 0);
    com_neg_mask = com_mask & (r_vec < 0);

    sumposC_train = sum(Xtrain_z(com_pos_mask,:),1)';
    sumnegC_train = sum(Xtrain_z(com_neg_mask,:),1)';

    sumposC_test = sum(Xtest_z(com_pos_mask));
    sumnegC_test = sum(Xtest_z(com_neg_mask));

    % ---- models ----
    mdl_com = fitlm([sumposC_train,sumnegC_train], ytrain, 'Intercept', true);
    b_com = mdl_com.Coefficients.Estimate;
    b_com_all(leftout,:) = [b_com(2), b_com(3), b_com(1)];

    mdl_pos = fitlm(sumpos_train, ytrain, 'Intercept', true);
    b_pos = mdl_pos.Coefficients.Estimate;
    b_pos_all(leftout,:) = [b_pos(2), b_pos(1)];

    mdl_neg = fitlm(sumneg_train, ytrain, 'Intercept', true);
    b_neg = mdl_neg.Coefficients.Estimate;
    b_neg_all(leftout,:) = [b_neg(2), b_neg(1)];

    pred_com(leftout) = b_com(2)*sumposC_test + b_com(3)*sumnegC_test + b_com(1);
    pred_pos(leftout) = b_pos(2)*sumpos_test + b_pos(1);
    pred_neg(leftout) = b_neg(2)*sumneg_test + b_neg(1);
end

% -------------------- Evaluation --------------------
[r_pos, p_pos] = corr(pred_pos, all_behav);
[r_neg, p_neg] = corr(pred_neg, all_behav);
[r_com, p_com] = corr(pred_com, all_behav);

se_pos = (pred_pos - all_behav).^2;
se_neg = (pred_neg - all_behav).^2;
se_com = (pred_com - all_behav).^2;

mse_pos = mean(se_pos);
mse_neg = mean(se_neg);
mse_com = mean(se_com);

cof_pos = regress(all_behav, [pred_pos, ones(nSubj,1)]);
cof_neg = regress(all_behav, [pred_neg, ones(nSubj,1)]);
cof_com = regress(all_behav, [pred_com, ones(nSubj,1)]);

S.pred = struct('pos', pred_pos, 'neg', pred_neg, 'com', pred_com);

S.eval = struct();
S.eval.r_pos = r_pos; S.eval.p_pos = p_pos;
S.eval.r_neg = r_neg; S.eval.p_neg = p_neg;
S.eval.r_com = r_com; S.eval.p_com = p_com;
S.eval.mse_pos = mse_pos;
S.eval.mse_neg = mse_neg;
S.eval.mse_com = mse_com;
S.eval.se_pos = se_pos;
S.eval.se_neg = se_neg;
S.eval.se_com = se_com;

S.model = struct();
S.model.b_pos_perfold = b_pos_all;
S.model.b_neg_perfold = b_neg_all;
S.model.b_com_perfold = b_com_all;
S.model.calib_pos = cof_pos;
S.model.calib_neg = cof_neg;
S.model.calib_com = cof_com;

S.debug = struct();
S.debug.pos_mask = pos_masks;
S.debug.neg_mask = neg_masks;
S.debug.com_mask = com_masks;
S.debug.n_pos = n_pos;
S.debug.n_neg = n_neg;
S.debug.n_com = n_com;
S.debug.utMask = utMask;
S.debug.fc_feat_idx = feat_idx;
S.debug.featureBlocks = S.meta.featureBlocks;

end

% ====================== helper ======================
function mask = topk_mask_by_p(p_vec, candidate_mask, K)
mask = false(size(p_vec));
idx = find(candidate_mask);
if isempty(idx) || K <= 0
    return;
end
[~, ord] = sort(p_vec(idx), 'ascend', 'MissingPlacement', 'last');
k = min(K, numel(idx));
pick = idx(ord(1:k));
mask(pick) = true;
end
