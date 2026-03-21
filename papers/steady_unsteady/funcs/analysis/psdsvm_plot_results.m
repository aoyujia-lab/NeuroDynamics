function [fig, roc] = a_psdsvm_plot_results(S, save_path, prefix)
% a_psdsvm_plot_results
% Plot ROC curves and confusion matrix from result struct S.
% Also OUTPUT ROC curve data (raw + smoothed) for later use.
%
% INPUT
%   S         : output struct from a_psdsvm_anova_range (or compatible)
%   save_path : (optional) folder to save figures
%   prefix    : (optional) filename prefix, e.g. 'range2'
%
% OUTPUT
%   fig : struct with figure handles: fig.roc, fig.cm
%   roc : struct containing ROC curves:
%         roc.X{c}, roc.Y{c}       : raw ROC points for class c (one-vs-rest)
%         roc.X_common             : common grid for smoothing (column vector)
%         roc.Y_smooth(:,c)        : smoothed ROC on common grid
%         roc.AUC(c)               : AUC per class (if available; otherwise computed)
%         roc.nCond                : number of classes
%         roc.legend               : legend strings

    if nargin < 2, save_path = ''; end
    if nargin < 3, prefix = ''; end

    y      = S.data.y(:);
    scores = S.cv.scores;
    y_pred = S.cv.y_pred(:);

    nCond = S.meta.nCond;

    % ---------- ROC data container ----------
    roc = struct();
    roc.nCond = nCond;
    roc.X = cell(1, nCond);
    roc.Y = cell(1, nCond);

    roc.X_common = linspace(0, 1, 100)';           % common grid
    roc.Y_smooth = nan(numel(roc.X_common), nCond);

    % AUC: prefer existing, otherwise compute
    if isfield(S, 'metrics') && isfield(S.metrics, 'AUC') && numel(S.metrics.AUC) == nCond
        roc.AUC = S.metrics.AUC(:)';
    else
        roc.AUC = nan(1, nCond);
    end

    fig = struct();

    % ---------- ROC plot ----------
    fig.roc = figure; hold on;
    for c = 1:nCond
        % One-vs-rest ROC: y as labels, posclass = c
        [Xroc, Yroc, ~, AUCc] = perfcurve(y, scores(:, c), c);

        roc.X{c} = Xroc(:);
        roc.Y{c} = Yroc(:);

        % fill AUC if missing
        if isnan(roc.AUC(c))
            roc.AUC(c) = AUCc;
        end

        % make X unique & sorted for stable interpolation
        [Xu, iu] = unique(roc.X{c}, 'stable');
        Yu = roc.Y{c}(iu);

        % perfcurve should already be monotonic in X, but enforce sorting just in case
        [Xu, ord] = sort(Xu);
        Yu = Yu(ord);

        roc.Y_smooth(:, c) = interp1(Xu, Yu, roc.X_common, 'linear', 'extrap');

        plot(Xroc, Yroc, 'LineWidth', 2);
    end
    plot([0 1], [0 1], '--', 'LineWidth', 1); % chance line

    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('ROC (multi-class, one-vs-rest)');
    grid on;

    roc.legend = arrayfun(@(c) sprintf('Class %d (AUC=%.3f)', c, roc.AUC(c)), ...
        1:nCond, 'UniformOutput', false);
    legend(roc.legend, 'Location', 'Best');
    hold off;

    % ---------- Confusion Matrix ----------
    fig.cm = figure;
    cm = confusionchart(y, y_pred, 'RowSummary', 'row-normalized');
    cm.Title = '';
    cm.XLabel = '';
    cm.YLabel = '';
    cm.FontSize = 18;
    set(gcf, 'Position', [100, 100, 800, 520]);

    % ---------- save ----------
    if ~isempty(save_path)
        if ~exist(save_path, 'dir'), mkdir(save_path); end

        if isempty(prefix)
            roc_name = 'roc.png';
            cm_name  = 'cm.png';
        else
            roc_name = sprintf('%s_roc.png', prefix);
            cm_name  = sprintf('%s_cm.png',  prefix);
        end

        try
            saveas(fig.roc, fullfile(save_path, roc_name));
            saveas(fig.cm,  fullfile(save_path, cm_name));
        catch
            warning('Failed to save figures.');
        end
    end
end
