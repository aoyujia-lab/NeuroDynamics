function [fig, roc] = psdsvm_plot_results(S, save_path, prefix)
%PSDSVM_PLOT_RESULTS Plot ROC curves and confusion matrix from SVM results.

if nargin < 2 || isempty(save_path)
    save_path = '';
end
if nargin < 3
    prefix = '';
end

y = S.data.y(:);
y_pred = S.cv.y_pred(:);
scores = S.cv.scores;
nCond = S.meta.nCond;

roc = struct();
roc.nCond = nCond;
roc.X = cell(1, nCond);
roc.Y = cell(1, nCond);
roc.X_common = linspace(0, 1, 100)';
roc.Y_smooth = nan(numel(roc.X_common), nCond);

if isfield(S, 'metrics') && isfield(S.metrics, 'AUC') && numel(S.metrics.AUC) == nCond
    roc.AUC = S.metrics.AUC(:)';
else
    roc.AUC = nan(1, nCond);
end

fig = struct();
fig.roc = figure;
hold on;
for c = 1:nCond
    [x_roc, y_roc, ~, auc_c] = perfcurve(y, scores(:, c), c);
    roc.X{c} = x_roc(:);
    roc.Y{c} = y_roc(:);

    if isnan(roc.AUC(c))
        roc.AUC(c) = auc_c;
    end

    [x_unique, unique_idx] = unique(roc.X{c}, 'stable');
    y_unique = roc.Y{c}(unique_idx);
    [x_unique, order] = sort(x_unique);
    y_unique = y_unique(order);
    roc.Y_smooth(:, c) = interp1(x_unique, y_unique, roc.X_common, 'linear', 'extrap');

    plot(x_roc, y_roc, 'LineWidth', 2);
end
plot([0 1], [0 1], '--', 'LineWidth', 1);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC (one-vs-rest)');
grid on;
roc.legend = arrayfun(@(c) sprintf('Class %d (AUC=%.3f)', c, roc.AUC(c)), 1:nCond, 'UniformOutput', false);
legend(roc.legend, 'Location', 'Best');
hold off;

fig.cm = figure;
cm = confusionchart(y, y_pred, 'RowSummary', 'row-normalized');
cm.Title = '';
cm.XLabel = '';
cm.YLabel = '';
cm.FontSize = 18;
set(gcf, 'Position', [100, 100, 800, 520]);

if ~isempty(save_path)
    if ~exist(save_path, 'dir')
        mkdir(save_path);
    end

    if isempty(prefix)
        roc_name = 'roc.png';
        cm_name = 'cm.png';
    else
        roc_name = sprintf('%s_roc.png', prefix);
        cm_name = sprintf('%s_cm.png', prefix);
    end

    saveas(fig.roc, fullfile(save_path, roc_name));
    saveas(fig.cm, fullfile(save_path, cm_name));
end
end
