function T = compare_steady_unsteady_paired_ttest(R)
% Compare steady vs unsteady using paired t-tests only,
% and plot one bar figure for each metric.
%
% RT metrics are based on correct trials only.
% ACC mean is based on all trials.

    rows = {};

    % ===== Collect metrics dynamically =====
    metrics = {};

    % RT mean
    metrics(end+1, :) = {'RT mean', R.steady.all.mean(:), R.unsteady.all.mean(:)};

    % RT SD
    metrics(end+1, :) = {'RT SD', R.steady.all.sd(:), R.unsteady.all.sd(:)};

    % RT CV
    steady_rt_cv   = R.steady.all.sd(:) ./ R.steady.all.mean(:);
    unsteady_rt_cv = R.unsteady.all.sd(:) ./ R.unsteady.all.mean(:);
    metrics(end+1, :) = {'RT CV', steady_rt_cv, unsteady_rt_cv};

    % RT entropy or SampEn
    if isfield(R.steady.all, 'entropy') && isfield(R.unsteady.all, 'entropy')
        metrics(end+1, :) = {'RT entropy', R.steady.all.entropy(:), R.unsteady.all.entropy(:)};
    elseif isfield(R.steady.all, 'sampen') && isfield(R.unsteady.all, 'sampen')
        metrics(end+1, :) = {'RT SampEn', R.steady.all.sampen(:), R.unsteady.all.sampen(:)};
    end

    % ACC mean
    metrics(end+1, :) = {'ACC mean', R.steady.acc(:), R.unsteady.acc(:)};

    % ACC SD (optional)
    if isfield(R.steady, 'acc_sd') && isfield(R.unsteady, 'acc_sd')
        metrics(end+1, :) = {'ACC SD', R.steady.acc_sd(:), R.unsteady.acc_sd(:)};
    end

    % ===== Run paired t-tests and plot =====
    for i = 1:size(metrics, 1)
        metric_name = metrics{i, 1};
        x = metrics{i, 2};   % steady
        y = metrics{i, 3};   % unsteady

        rows(end+1, :) = one_metric_ttest(metric_name, x, y);
        plot_metric_bar(metric_name, x, y);
    end

    % ===== Output table =====
    T = cell2table(rows, 'VariableNames', ...
        {'Metric', ...
         'Steady_Mean', 'Steady_SD_acrossSubj', ...
         'Unsteady_Mean', 'Unsteady_SD_acrossSubj', ...
         'Diff_SteadyMinusUnsteady_Mean', 'Diff_SteadyMinusUnsteady_SD', ...
         'N', 't', 'df', 'p', 'Cohens_dz'});
end

function row = one_metric_ttest(metric_name, x, y)
    x = x(:);
    y = y(:);

    valid = ~isnan(x) & ~isnan(y);
    x = x(valid);
    y = y(valid);

    d = x - y;
    n = numel(d);

    [~, p, ~, stats] = ttest(x, y);

    if std(d) > 0
        dz = mean(d) / std(d);   % Cohen's dz for paired samples
    else
        dz = NaN;
    end

    row = {metric_name, ...
           mean(x), std(x), ...
           mean(y), std(y), ...
           mean(d), std(d), ...
           n, stats.tstat, stats.df, p, dz};
end

function plot_metric_bar(metric_name, x, y)
    x = x(:);
    y = y(:);

    valid = ~isnan(x) & ~isnan(y);
    x = x(valid);
    y = y(valid);

    n = numel(x);

    mx = mean(x);
    my = mean(y);

    semx = std(x) / sqrt(n);
    semy = std(y) / sqrt(n);

    [~, p, ~, stats] = ttest(x, y);

    figure('Color', 'w', 'Name', metric_name);
    hold on;

    % bars
    bar(1, mx, 0.6, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'k');
    bar(2, my, 0.6, 'FaceColor', [0.4 0.4 0.4], 'EdgeColor', 'k');

    % error bars
    errorbar([1 2], [mx my], [semx semy], 'k', ...
        'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 10);

    % paired subject lines
    for i = 1:n
        plot([1 2], [x(i) y(i)], '-', 'Color', [0.75 0.75 0.75], 'LineWidth', 0.8);
    end

    % subject points
    scatter(ones(n,1), x, 28, 'k', 'filled', ...
        'MarkerFaceAlpha', 0.75, 'MarkerEdgeAlpha', 0.75);
    scatter(2*ones(n,1), y, 28, 'k', 'filled', ...
        'MarkerFaceAlpha', 0.75, 'MarkerEdgeAlpha', 0.75);

    xlim([0.5 2.5]);
    set(gca, 'XTick', [1 2], 'XTickLabel', {'Steady', 'Unsteady'}, ...
        'FontSize', 12, 'Box', 'off', 'LineWidth', 1.2);

    ylabel(metric_name, 'FontSize', 12, 'Interpreter', 'none');
    title(sprintf('%s: t(%d)=%.3f, p=%.4f', metric_name, stats.df, stats.tstat, p), ...
        'FontSize', 12, 'Interpreter', 'none');

    hold off;
end