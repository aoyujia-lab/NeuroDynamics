function delong_tbl = pairwise_delong_multiclass(S_svm, svm_tags)
% Pairwise DeLong tests for multi-class results using one-vs-rest AUC.
% Condition index 0  = macro-average (stacked one-vs-rest DeLong).
% Condition index 1…nCond = per-class one-vs-rest.
% FDR correction is applied separately within each condition (incl. macro).

    nModel = numel(S_svm);
    nCond  = size(S_svm{1}.cv.scores, 2);
    delong_tbl = cell(nCond + 1, 1);

    allR  = struct([]);
    k_all = 0;

    % ── per-class one-vs-rest ─────────────────────────────────────────────
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

    % ── macro-average: stacked one-vs-rest DeLong ────────────────────────
    for i = 1:nModel-1
        for j = i+1:nModel
            y_stack  = [];
            si_stack = [];
            sj_stack = [];
            for c = 1:nCond
                y_stack  = [y_stack;  double(S_svm{i}.data.y == c)]; %#ok<AGROW>
                si_stack = [si_stack; S_svm{i}.cv.scores(:, c)];     %#ok<AGROW>
                sj_stack = [sj_stack; S_svm{j}.cv.scores(:, c)];     %#ok<AGROW>
            end

            auc_macro_i = S_svm{i}.metrics.macro_AUC;
            auc_macro_j = S_svm{j}.metrics.macro_AUC;

            [~, p_macro, se_i_macro, se_j_macro] = ...
                delong(si_stack, sj_stack, y_stack);

            k_all = k_all + 1;
            allR(k_all).Condition = 0;
            allR(k_all).Model1    = i;
            allR(k_all).Model2    = j;
            allR(k_all).Tag1      = string(svm_tags{i});
            allR(k_all).Tag2      = string(svm_tags{j});
            allR(k_all).AUC1      = auc_macro_i;
            allR(k_all).AUC2      = auc_macro_j;
            allR(k_all).DeltaAUC  = auc_macro_i - auc_macro_j;
            allR(k_all).PValue    = p_macro;
            allR(k_all).SE1       = se_i_macro;
            allR(k_all).SE2       = se_j_macro;
        end
    end

    % ── FDR 分组校正：每个 condition（含 macro=0）单独校正 ───────────────
    allTbl       = struct2table(allR);
    allTbl.P_FDR = nan(height(allTbl), 1);

    for c = unique(allTbl.Condition)'
        idx = allTbl.Condition == c;
        allTbl.P_FDR(idx) = mafdr(allTbl.PValue(idx), 'BHFDR', true);
    end

    % ── split output: slot 1…nCond = per-class, slot nCond+1 = macro ─────
    for c = 1:nCond
        delong_tbl{c} = allTbl(allTbl.Condition == c, :);
    end
    delong_tbl{nCond + 1} = allTbl(allTbl.Condition == 0, :);
end