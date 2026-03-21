function delong_tbl = pairwise_delong_multiclass(S_svm, svm_tags)
% Pairwise DeLong tests for multi-class results using one-vs-rest AUC
%
% INPUT
%   S_svm     : cell array, each cell is output of psdsvm_anova_range
%   svm_tags  : cell array of model names/tags
%
% OUTPUT
%   delong_tbl : cell array, one table per condition
%
% NOTE
%   FDR correction is applied across ALL pairwise tests from ALL conditions
%   together (e.g., 18 tests in total if 4 models x 3 conditions).

    nModel = numel(S_svm);
    nCond  = size(S_svm{1}.cv.scores, 2);

    delong_tbl = cell(nCond, 1);

    % collect all results first
    allR = struct([]);
    k_all = 0;

    for c = 1:nCond
        for i = 1:nModel-1
            for j = i+1:nModel

                assert(isequal(S_svm{i}.data.y, S_svm{j}.data.y), ...
                    'Sample order mismatch between models %d and %d.', i, j);

                y_true  = double(S_svm{i}.data.y == c);
                score_i = S_svm{i}.cv.scores(:, c);
                score_j = S_svm{j}.cv.scores(:, c);

                [~,~,~,auc_i] = perfcurve(y_true, score_i, 1);
                [~,~,~,auc_j] = perfcurve(y_true, score_j, 1);

                [~, p, se_i, se_j] = delong(score_i, score_j, y_true);

                k_all = k_all + 1;
                allR(k_all).Condition = c;
                allR(k_all).Model1    = i;
                allR(k_all).Model2    = j;
                allR(k_all).Tag1      = string(svm_tags{i});
                allR(k_all).Tag2      = string(svm_tags{j});
                allR(k_all).AUC1      = auc_i;
                allR(k_all).AUC2      = auc_j;
                allR(k_all).DeltaAUC  = auc_i - auc_j;
                allR(k_all).PValue    = p;
                allR(k_all).SE1       = se_i;
                allR(k_all).SE2       = se_j;
            end
        end
    end

    % convert to one big table
    allTbl = struct2table(allR);

    % FDR across ALL comparisons together
    allTbl.P_FDR = mafdr(allTbl.PValue, 'BHFDR', true);

    % split back into one table per condition
    for c = 1:nCond
        delong_tbl{c} = allTbl(allTbl.Condition == c, :);
    end
end