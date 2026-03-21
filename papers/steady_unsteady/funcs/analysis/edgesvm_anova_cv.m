function S = edgesvm_anova_cv(conn4d, C)
% edgesvm_anova_cv
% Train/Evaluate multi-class ECOC SVM with within-fold RM-ANOVA feature selection
% using edge features from connectivity matrices.
%
% INPUT
%   conn4d : [nROI x nROI x nSub x nCond]
%   C      : config struct, expects:
%       C.ml.alpha      (pThresh; if >=1 -> no feature selection)
%       C.ml.Kfallback  (fallback top-K pvals if none selected; if 0 -> disable)
%       C.ml.cvMode     'LOO' (default) or 'LOSO'
%       C.ml.svmlearner 'default' or a templateSVM object
%
% OUTPUT (struct S)
%   S.meta, S.data, S.cv, S.metrics

% ---- unpack config (match psdsvm_anova_range style) ----
pThresh    = C.ml.alpha;
Kfallback  = C.ml.Kfallback;
cvMode     = C.ml.cvMode;
svmLearner = C.ml.svmlearner;

% ---- checks ----
assert(ndims(conn4d) == 4, 'conn4d must be [nROI x nROI x nSub x nCond].');
[nROI, nROI2, nSub, nCond] = size(conn4d);
assert(nROI == nROI2, 'First two dims must be square (nROI x nROI).');
assert(nCond >= 2, 'Need at least 2 conditions.');

% ---- vectorize upper triangle as features ----
utMask = triu(true(nROI), 1);
nFeat  = nnz(utMask);

% ---- build X, y (like psdsvm: stack conditions) ----
% samples are ordered by condition blocks: all subjects for cond1, then cond2, ...
num_samples = nSub * nCond;
X = zeros(num_samples, nFeat, 'double');
y = zeros(num_samples, 1);

idx = 0;
for c = 1:nCond
    for s = 1:nSub
        idx = idx + 1;
        mat = conn4d(:,:,s,c);
        X(idx, :) = double(mat(utMask)).';
        y(idx)    = c;
    end
end

% 与这个顺序一致：cond block + subj
subj_id = repmat((1:nSub)', nCond, 1);
cond_id = repelem((1:nCond)', nSub);


% ---- define folds ----
switch upper(cvMode)
    case 'LOO'
        nFold = num_samples;
        fold_test_idx = num2cell(1:num_samples);

    case 'LOSO'
        nFold = nSub;
        fold_test_idx = cell(nFold, 1);
        for s = 1:nSub
            fold_test_idx{s} = find(subj_id == s); % all conditions for that subject
        end

    otherwise
        error('cvMode must be ''LOO'' or ''LOSO''.');
end

% ---- parse learner (same logic as psdsvm) ----
if ischar(svmLearner) || isstring(svmLearner)
    key = upper(string(svmLearner));
    switch key
        case "DEFAULT"
            learnerSpec = 'svm';
            learnerName = 'default_svm';
        otherwise
            error('svmLearner must be ''default'' or a templateSVM object.');
    end
else
    learnerSpec = svmLearner;      % templateSVM object
    learnerName = class(svmLearner);
end

% ---- storage ----
y_pred   = zeros(num_samples, 1);
scores   = zeros(num_samples, nCond);
featMask = false(nFold, nFeat);

% ---- CV loop ----
for f = 1:nFold
    test_idx = fold_test_idx{f};
    train_mask = true(num_samples, 1);
    train_mask(test_idx) = false;

    X_train = X(train_mask, :);
    y_train = y(train_mask);

    train_subj = subj_id(train_mask);
    train_cond = cond_id(train_mask);

    % ===== feature selection on TRAIN only =====
    if pThresh >= 1
        selFeat = true(nFeat, 1);
    else
        subList = unique(train_subj, 'stable');
        nSubTr  = numel(subList);

        % only keep subjects with all conditions present in training
        hasAll = false(nSubTr, 1);
        for si = 1:nSubTr
            sID = subList(si);
            hasAll(si) = all(ismember(1:nCond, unique(train_cond(train_subj == sID))));
        end
        subUse   = subList(hasAll);
        nSubUse  = numel(subUse);

        if nSubUse < 3
            selFeat = true(nFeat, 1);
        else
            % Xrm: [nSubUse x nCond x nFeat]
            Xrm = zeros(nSubUse, nCond, nFeat, 'double');
            for si = 1:nSubUse
                sID = subUse(si);
                for c = 1:nCond
                    row = (train_subj == sID) & (train_cond == c);
                    % should be exactly one sample for each (sub,cond)
                    Xrm(si, c, :) = X_train(row, :);
                end
            end

            grandMean = mean(Xrm, [1 2], 'omitnan');   % 1x1xnFeat
            subjMean  = mean(Xrm, 2, 'omitnan');       % nSubUse x1xnFeat
            condMean  = mean(Xrm, 1, 'omitnan');       % 1xnCondxnFeat

            SS_cond = nSubUse * sum((condMean - grandMean).^2, 2);
            SS_err  = sum((Xrm - subjMean - condMean + grandMean).^2, [1 2]);

            df1 = nCond - 1;
            df2 = df1 * (nSubUse - 1);

            Fval = (SS_cond/df1) ./ (SS_err/df2);
            pval = 1 - fcdf(Fval, df1, df2);
            pval = squeeze(pval);  % [nFeat x 1]

            selFeat = (pval < pThresh);

            if ~any(selFeat) && Kfallback > 0
                [~, ord] = sort(pval, 'ascend');
                K = min(Kfallback, numel(ord));
                selFeat(ord(1:K)) = true;
            end
        end
    end

    featMask(f, :) = selFeat;

    % ===== train/test =====
    mdl = fitcecoc(X_train(:, selFeat), y_train, ...
        'Learners', learnerSpec, ...
        'Coding', 'onevsone', ...
        'ClassNames', 1:nCond);

    [yhat, sc] = predict(mdl, X(test_idx, selFeat));
    y_pred(test_idx)   = yhat;
    scores(test_idx, :) = sc;
end

% ---- metrics (same style as psdsvm) ----
AUC = nan(nCond, 1);
for c = 1:nCond
    posClass = (y == c);
    [~, ~, ~, AUC(c)] = perfcurve(posClass, scores(:, c), true);
end
acc = mean(y_pred == y);

% ---- pack ----
S = struct();

S.meta = struct( ...
    'nROI', nROI, 'nFeat', nFeat, 'nSub', nSub, 'nCond', nCond, ...
    'cvMode', cvMode, 'pThresh', pThresh, 'Kfallback', Kfallback, ...
    'svmLearner', char(learnerName), ...
    'featureType', 'upper-triangular edges');

S.data = struct( ...
    'X', X, 'y', y, ...
    'subj_id', subj_id, 'cond_id', cond_id, ...
    'utMask', utMask);

S.cv = struct( ...
    'y_pred', y_pred, ...
    'scores', scores, ...
    'featMask', featMask);

S.metrics = struct( ...
    'AUC', AUC, ...
    'acc', acc);
end
