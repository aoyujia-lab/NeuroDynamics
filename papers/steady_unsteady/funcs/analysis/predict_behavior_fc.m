function S = predict_behavior_fc(all_vects, all_behav, cov, C)
% INPUT
%   all_vects : [nROI x nROI x nSubj]
%   all_behav : [nSubj x 1]
%   cov       : [nSubj x nCov] (required when fs_option = 4)
%   C         : config
%
% OUTPUT (struct S)
%   S.meta.*, S.pred.*, S.eval.*, S.model.*, S.debug.*

% -------------------- Input --------------------
fs_option = C.ml.corr_option;
thresh    = C.ml.alpha;
Kfallback = C.ml.Kfallback;
use_fallback = (Kfallback > 0);

assert(ndims(all_vects) == 3, 'all_vects must be [nROI x nROI x nSubj].');
[nROI, nROI2, nSubj] = size(all_vects);
assert(nROI == nROI2, 'all_vects first two dims must be square (nROI x nROI).');
assert(size(all_behav,1) == nSubj, 'all_behav must be [nSubj x 1].');

% ---- vectorize to features (upper triangle, exclude diagonal) ----
utMask = triu(true(nROI), 1);
feat_idx = find(utMask);
nFeat = numel(feat_idx);

% Xfeat: [nFeat x nSubj]
Xfeat = zeros(nFeat, nSubj, 'double');
for s = 1:nSubj
    mat = all_vects(:,:,s);
    Xfeat(:,s) = double(mat(feat_idx));
end

% -------------------- Meta --------------------
S = struct();
S.meta = struct();
S.meta.nROI      = nROI;
S.meta.nSubj     = nSubj;
S.meta.thresh    = thresh;
S.meta.fs_option = fs_option;
S.meta.has_cov   = (fs_option == 4);
S.meta.Kfallback = Kfallback;
S.meta.nFeat     = nFeat;
S.meta.featureType = 'upper-triangular edges';
S.meta.note      = 'LOO CPM on edge-level features (threshold + fallback-to-topK if needed)';

switch fs_option
    case 1, sel_msg = 'Pearson correlation';
    case 2, sel_msg = 'Spearman correlation';
    case 3, sel_msg = 'Robust regression';
    case 4, sel_msg = 'Partial correlation';
    otherwise, error('Unknown fs_option.');
end
fprintf('Feature selection: %s\n', sel_msg);
fprintf('Leave-one-out CV (%d subjects)\n', nSubj);
fprintf('thresh = %.4g, Kfallback = %d (0=off)\n', thresh, Kfallback);

% -------------------- Allocate --------------------
pred_pos = zeros(nSubj,1);
pred_neg = zeros(nSubj,1);
pred_com = zeros(nSubj,1);

b_pos_all = nan(nSubj,2);   % [slope intercept] per fold
b_neg_all = nan(nSubj,2);
b_com_all = nan(nSubj,3);   % [b_pos b_neg intercept] per fold

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

    Xtrain = Xfeat(:, train_idx);   % [nFeat x (nSubj-1)]
    ytrain = all_behav(train_idx);  % [(nSubj-1) x 1]
    nTrain = sum(train_idx);

    % ---- feature-wise association on TRAIN ----
    switch fs_option
        case 1
            [r_vec, p_vec] = corr(Xtrain', ytrain);  % r_vec: [nFeat x 1]
        case 2
            [r_vec, p_vec] = corr(Xtrain', ytrain, 'type', 'Spearman');
        case 3
            warning('off');
            r_vec = zeros(nFeat,1);
            p_vec = zeros(nFeat,1);
            for i = 1:nFeat
                [~, stats] = robustfit(Xtrain(i,:)', ytrain);
                cur_t = stats.t(2);
                df = nTrain - 2;
                r_vec(i) = sign(cur_t) * sqrt(cur_t^2 / (df + cur_t^2));
                p_vec(i) = 2 * (1 - tcdf(abs(cur_t), df));
            end
            warning('on');
        case 4
            cov_train = cov(train_idx, :);
            [r_vec, p_vec] = partialcorr(Xtrain', ytrain, cov_train);
    end

    % ---- masks with threshold + fallback (separately) ----
    % POS candidates: r>0
    pos_sig = (r_vec > 0) & (p_vec < thresh);
    if use_fallback && nnz(pos_sig) < Kfallback
        pos_mask = topk_mask_by_p(p_vec, (r_vec > 0), Kfallback);
    else
        pos_mask = pos_sig;
    end

    % NEG candidates: r<0
    neg_sig = (r_vec < 0) & (p_vec < thresh);
    if use_fallback && nnz(neg_sig) < Kfallback
        neg_mask = topk_mask_by_p(p_vec, (r_vec < 0), Kfallback);
    else
        neg_mask = neg_sig;
    end

    % COM candidates: all features (p<thresh), fallback to topK overall if too few
    com_sig = (p_vec < thresh);
    if use_fallback && nnz(com_sig) < Kfallback
        com_mask = topk_mask_by_p(p_vec, true(nFeat,1), Kfallback);
    else
        com_mask = com_sig;
    end

    pos_masks(:, leftout) = pos_mask;
    neg_masks(:, leftout) = neg_mask;
    com_masks(:, leftout) = com_mask;

    n_pos(leftout) = nnz(pos_mask);
    n_neg(leftout) = nnz(neg_mask);
    n_com(leftout) = nnz(com_mask);

    % ---- sum features ----
    sumpos_train = sum(Xtrain(pos_mask, :), 1)'; % [nTrain x 1]
    sumneg_train = sum(Xtrain(neg_mask, :), 1)';

    sumpos_test = sum(Xfeat(pos_mask, leftout));
    sumneg_test = sum(Xfeat(neg_mask, leftout));

    % combined: within com_mask, split by sign (consistent with CPM pos/neg idea)
    com_pos_mask = com_mask & (r_vec > 0);
    com_neg_mask = com_mask & (r_vec < 0);

    sumposC_train = sum(Xtrain(com_pos_mask, :), 1)'; % [nTrain x 1]
    sumnegC_train = sum(Xtrain(com_neg_mask, :), 1)';

    sumposC_test = sum(Xfeat(com_pos_mask, leftout));
    sumnegC_test = sum(Xfeat(com_neg_mask, leftout));

    % ---- build models on TRAIN ----
    mdl_com = fitlm([sumposC_train, sumnegC_train], ytrain, 'Intercept', true);
    b_com = mdl_com.Coefficients.Estimate; % [intercept; b_pos; b_neg]
    b_com_all(leftout,:) = [b_com(2), b_com(3), b_com(1)];

    mdl_pos = fitlm(sumpos_train, ytrain, 'Intercept', true);
    b_pos = mdl_pos.Coefficients.Estimate; % [intercept; slope]
    b_pos_all(leftout,:) = [b_pos(2), b_pos(1)];

    mdl_neg = fitlm(sumneg_train, ytrain, 'Intercept', true);
    b_neg = mdl_neg.Coefficients.Estimate; % [intercept; slope]
    b_neg_all(leftout,:) = [b_neg(2), b_neg(1)];

    % ---- predict TEST ----
    pred_com(leftout) = b_com(2)*sumposC_test + b_com(3)*sumnegC_test + b_com(1);
    pred_pos(leftout) = b_pos(2)*sumpos_test  + b_pos(1);
    pred_neg(leftout) = b_neg(2)*sumneg_test  + b_neg(1);
end

% -------------------- Evaluation --------------------
[r_pos, p_pos] = corr(pred_pos, all_behav);
[r_neg, p_neg] = corr(pred_neg, all_behav);
[r_com, p_com] = corr(pred_com, all_behav);

% --- per-subject squared error (single-point MSE per subject) ---
se_pos = (pred_pos - all_behav).^2;   % [nSubj x 1]
se_neg = (pred_neg - all_behav).^2;
se_com = (pred_com - all_behav).^2;

% --- overall MSE ---
mse_pos = mean(se_pos, 'omitnan');
mse_neg = mean(se_neg, 'omitnan');
mse_com = mean(se_com, 'omitnan');

% optional calibration
cof_pos = regress(all_behav, [pred_pos, ones(nSubj,1)]);
cof_neg = regress(all_behav, [pred_neg, ones(nSubj,1)]);
cof_com = regress(all_behav, [pred_com, ones(nSubj,1)]);

% -------------------- Pack outputs --------------------
S.pred = struct('pos', pred_pos, 'neg', pred_neg, 'com', pred_com);

S.eval = struct();
S.eval.r_pos = r_pos; S.eval.p_pos = p_pos;
S.eval.r_neg = r_neg; S.eval.p_neg = p_neg;
S.eval.r_com = r_com; S.eval.p_com = p_com;
S.eval.mse_pos = mse_pos;
S.eval.mse_neg = mse_neg;
S.eval.mse_com = mse_com;

S.eval.se_pos = se_pos;   % per-subject squared error
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

% extra: mapping back to ROI pairs if you need later
S.debug.utMask = utMask;
S.debug.feat_idx = feat_idx;

end

% ====================== helper ======================
function mask = topk_mask_by_p(p_vec, candidate_mask, K)
% pick up to K indices with smallest p among candidate_mask
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
