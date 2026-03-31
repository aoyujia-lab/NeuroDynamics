function T = compare_steady_unsteady_signedrank(R)
% Compare steady vs unsteady using paired Wilcoxon signed-rank tests,
% and plot one figure for each metric using median +/- IQR.
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

    % ===== Run signed-rank tests and plot =====
    for i = 1:size(metrics, 1)
        metric_name = metrics{i, 1};
        x = metrics{i, 2};   % steady
        y = metrics{i, 3};   % unsteady

        rows(end+1, :) = one_metric_signedrank(metric_name, x, y);
        plot_metric_rank(metric_name, x, y);
    end

    % ===== Output table =====
    T = cell2table(rows, 'VariableNames', ...
        {'Metric', ...
         'Steady_Median', 'Steady_IQR', ...
         'Unsteady_Median', 'Unsteady_IQR', ...
         'Diff_SteadyMinusUnsteady_Median', 'Diff_SteadyMinusUnsteady_IQR', ...
         'N', 'SignedRank_W', 'p', 'RankBiserial_r'});
end


function row = one_metric_signedrank(metric_name, x, y)
    x = x(:);
    y = y(:);

    valid = ~isnan(x) & ~isnan(y);
    x = x(valid);
    y = y(valid);

    d = x - y;
    n = numel(d);

    % Wilcoxon signed-rank test
    [p, ~, stats] = signrank(x, y);

    % Effect size: matched-pairs rank-biserial correlation
    r_rb = compute_rank_biserial(d);

    row = {metric_name, ...
           median(x), iqr(x), ...
           median(y), iqr(y), ...
           median(d), iqr(d), ...
           n, stats.signedrank, p, r_rb};
end


function plot_metric_rank(metric_name, x, y)
    x = x(:);
    y = y(:);

    valid = ~isnan(x) & ~isnan(y);
    x = x(valid);
    y = y(valid);

    n = numel(x);

    medx = median(x);
    medy = median(y);

    qx = prctile(x, [25 75]);
    qy = prctile(y, [25 75]);

    [p, ~, stats] = signrank(x, y);

    figure('Color', 'w', 'Name', metric_name);
    hold on;

    % paired subject lines
    for i = 1:n
        plot([1 2], [x(i) y(i)], '-', 'Color', [0.82 0.82 0.82], 'LineWidth', 0.8);
    end

    % subject points
    scatter(ones(n,1), x, 30, 'k', 'filled', ...
        'MarkerFaceAlpha', 0.75, 'MarkerEdgeAlpha', 0.75);
    scatter(2*ones(n,1), y, 30, 'k', 'filled', ...
        'MarkerFaceAlpha', 0.75, 'MarkerEdgeAlpha', 0.75);

    % median points
    scatter(1, medx, 90, 'd', 'filled', ...
        'MarkerFaceColor', [0.3 0.3 0.3], 'MarkerEdgeColor', 'k');
    scatter(2, medy, 90, 'd', 'filled', ...
        'MarkerFaceColor', [0.3 0.3 0.3], 'MarkerEdgeColor', 'k');

    % IQR error bars
    errorbar(1, medx, medx-qx(1), qx(2)-medx, 'k', ...
        'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 10);
    errorbar(2, medy, medy-qy(1), qy(2)-medy, 'k', ...
        'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 10);

    xlim([0.5 2.5]);
    set(gca, 'XTick', [1 2], 'XTickLabel', {'Steady', 'Unsteady'}, ...
        'FontSize', 12, 'Box', 'off', 'LineWidth', 1.2);

    ylabel(metric_name, 'FontSize', 12, 'Interpreter', 'none');
    title(sprintf('%s: signed-rank W=%.3f, p=%.4f', ...
        metric_name, stats.signedrank, p), ...
        'FontSize', 12, 'Interpreter', 'none');

    hold off;
end


function r_rb = compute_rank_biserial(d)
% Matched-pairs rank-biserial correlation
% Based on signed ranks of non-zero differences.

    d = d(:);
    d = d(~isnan(d));
    d = d(d ~= 0);   % remove zero differences

    if isempty(d)
        r_rb = NaN;
        return;
    end

    abs_d = abs(d);
    ranks = tiedrank(abs_d);

    Wpos = sum(ranks(d > 0));
    Wneg = sum(ranks(d < 0));

    r_rb = (Wpos - Wneg) / (Wpos + Wneg);
end