function S = psdsvm_anova_range(band_psd, C)
% psdsvm_anova_range
% Train/Evaluate multi-class ECOC SVM with within-fold RM-ANOVA feature selection.
%
% INPUT
%   band_psd   : [nROI x nSub x nCond]  (single frequency range)
%   pThresh    : default 0.01; if >=1 -> no feature selection
%   Kfallback  : default 200; if 0 -> disable fallback
%   cvMode     : 'LOO' (default) or 'LOSO'
%   svmLearner : (optional) choose learner
%                - 'default' : use MATLAB default SVM learner (same as 'Learners','svm')
%                - 'linear'  : use templateSVM('linear','Standardize',true,'BoxConstraint',1)
%                - template  : pass a templateSVM object directly
%
% OUTPUT (struct S)
%   S.meta, S.data, S.cv, S.metrics


    pThresh = C.ml.alpha;
    Kfallback = C.ml.Kfallback;
    cvMode = C.ml.cvMode;
    svmLearner = C.ml.svmlearner;


    assert(ndims(band_psd) == 3, 'band_psd must be [nROI x nSub x nCond].');
    [nROI, nSub, nCond] = size(band_psd);
    assert(nCond >= 2, 'Need at least 2 conditions.');

    % ---------- build X, y ----------
    X = zeros(nSub*nCond, nROI, 'double');
    y = zeros(nSub*nCond, 1);
    row0 = 0;
    for c = 1:nCond
        rows = (row0+1):(row0+nSub);
        X(rows,:) = double(band_psd(:,:,c)'); % [nSub x nROI]
        y(rows)   = c;
        row0 = row0 + nSub;
    end

    subj_id = repmat((1:nSub)', nCond, 1);
    cond_id = repelem((1:nCond)', nSub);
    num_samples = size(X,1);

    % ---------- define folds ----------
    switch upper(cvMode)
        case 'LOO'
            nFold = num_samples;
            fold_test_idx = num2cell(1:num_samples);
        case 'LOSO'
            nFold = nSub;
            fold_test_idx = cell(nFold,1);
            for s = 1:nSub
                fold_test_idx{s} = find(subj_id == s);
            end
        otherwise
            error('cvMode must be ''LOO'' or ''LOSO''.');
    end

    y_pred = zeros(num_samples,1);
    scores = zeros(num_samples, nCond);
    featMask = false(nFold, nROI);

    % ---------- parse learner ----------
    % learnerSpec will be used as value for fitcecoc(...,'Learners', learnerSpec)
    if ischar(svmLearner) || isstring(svmLearner)
        key = upper(string(svmLearner));
        switch key
            case "DEFAULT"
                learnerSpec = 'svm'; % MATLAB default SVM learner
                learnerName = 'default_svm';  
            otherwise
                error('svmLearner must be ''default'' or a templateSVM object.');
        end
    else
        % allow passing a templateSVM object directly
        learnerSpec = svmLearner;
        learnerName = class(svmLearner);
    end

    % ---------- CV loop ----------
    for f = 1:nFold
        test_idx = fold_test_idx{f};
        train_mask = true(num_samples,1);
        train_mask(test_idx) = false;

        X_train = X(train_mask,:);
        y_train = y(train_mask);

        train_subj = subj_id(train_mask);
        train_cond = cond_id(train_mask);

        % ----- feature selection on TRAIN only -----
        if pThresh >= 1
            selFeat = true(nROI,1);
        else
            subList = unique(train_subj, 'stable');
            nSubTr  = numel(subList);

            % only keep subjects with all conditions present in training
            hasAll = false(nSubTr,1);
            for si = 1:nSubTr
                sID = subList(si);
                hasAll(si) = all(ismember(1:nCond, unique(train_cond(train_subj==sID))));
            end
            subUse = subList(hasAll);
            nSubUse = numel(subUse);

            if nSubUse < 3
                selFeat = true(nROI,1);
            else
                Xrm = zeros(nSubUse, nCond, nROI, 'double');
                for si = 1:nSubUse
                    sID = subUse(si);
                    for c = 1:nCond
                        row = (train_subj==sID) & (train_cond==c);
                        Xrm(si,c,:) = X_train(row,:);
                    end
                end

                grandMean = mean(Xrm, [1 2], 'omitnan');   % 1x1xnROI
                subjMean  = mean(Xrm, 2, 'omitnan');       % nSubUse x1xnROI
                condMean  = mean(Xrm, 1, 'omitnan');       % 1xnCondxnROI

                SS_cond = nSubUse * sum((condMean - grandMean).^2, 2);
                SS_err  = sum((Xrm - subjMean - condMean + grandMean).^2, [1 2]);

                df1 = nCond - 1;
                df2 = df1 * (nSubUse - 1);

                Fval = (SS_cond/df1) ./ (SS_err/df2);
                pval = 1 - fcdf(Fval, df1, df2);
                pval = squeeze(pval);  % [nROI x 1]

                selFeat = (pval < pThresh);

                if ~any(selFeat) && Kfallback > 0
                    [~, ord] = sort(pval, 'ascend');
                    K = min(Kfallback, numel(ord));
                    selFeat(ord(1:K)) = true;
                end
            end
        end

        featMask(f,:) = selFeat;

        % ----- train / test -----
        mdl = fitcecoc(X_train(:,selFeat), y_train, ...
            'Learners', learnerSpec, ...
            'Coding', 'onevsone', ...
            'ClassNames', 1:nCond);

        [yhat, sc] = predict(mdl, X(test_idx, selFeat));
        y_pred(test_idx) = yhat;
        scores(test_idx,:) = sc;
    end

    % ---------- metrics ----------
    AUC = nan(nCond,1);
    for c = 1:nCond
        posClass = (y == c);
        [~,~,~,AUC(c)] = perfcurve(posClass, scores(:,c), true);
    end
    acc = mean(y_pred == y);

    % ---------- pack ----------
    S = struct();
    S.meta = struct('nROI',nROI,'nSub',nSub,'nCond',nCond, ...
                    'cvMode',cvMode,'pThresh',pThresh,'Kfallback',Kfallback, ...
                    'svmLearner', char(learnerName));

    S.data = struct('X',X,'y',y,'subj_id',subj_id,'cond_id',cond_id);
    S.cv = struct('y_pred',y_pred,'scores',scores,'featMask',featMask);
    S.metrics = struct('AUC',AUC,'acc',acc);
end
