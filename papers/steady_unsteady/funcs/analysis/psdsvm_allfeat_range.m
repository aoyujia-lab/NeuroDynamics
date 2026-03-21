function S = psdsvm_allfeat_range(band_psd, C)
% psdsvm_allfeat_range
% Train/Evaluate multi-class ECOC SVM using ALL features (no ANOVA selection).
%
% INPUT
%   band_psd   : [nROI x nSub x nCond]  (single frequency range)
%   cvMode     : 'LOO' (default) or 'LOSO'
%   svmLearner : (optional) choose learner
%                - 'default' : use MATLAB default SVM learner
%                - template  : pass a templateSVM object directly
%
% OUTPUT (struct S)
%   S.meta, S.data, S.cv, S.metrics

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
    featMask = true(nFold, nROI);   % all features used in every fold

    % ---------- parse learner ----------
    if ischar(svmLearner) || isstring(svmLearner)
        key = upper(string(svmLearner));
        switch key
            case "DEFAULT"
                learnerSpec = 'svm';   % MATLAB default SVM learner
                learnerName = 'default_svm';
            otherwise
                error('svmLearner must be ''default'' or a templateSVM object.');
        end
    else
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

        % ----- use all features -----
        selFeat = true(nROI,1);

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
                    'cvMode',cvMode, ...
                    'featureMode','all', ...
                    'svmLearner', char(learnerName));

    S.data = struct('X',X,'y',y,'subj_id',subj_id,'cond_id',cond_id);
    S.cv = struct('y_pred',y_pred,'scores',scores,'featMask',featMask);
    S.metrics = struct('AUC',AUC,'acc',acc);
end