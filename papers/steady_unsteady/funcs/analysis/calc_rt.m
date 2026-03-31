function R = calc_rt(P, C)
% CALC_RT Compute RT summary statistics for steady and unsteady tasks.
%
% RT metrics are computed on correct trials only.
% ACC is computed on all trials.
% RT SampEn is computed on the RT sequence of correct trials.

behavior_path = P.data.behav;
steady_files = dir(fullfile(behavior_path, '*_steady*'));
unsteady_files = dir(fullfile(behavior_path, '*_unsteady*'));

steady_files(C.data.excludesubjects) = [];
unsteady_files(C.data.excludesubjects) = [];

nSubj = numel(steady_files);

% default SampEn parameters
m = 2;
r = 0.2;
min_n_sampen = 50;

if isfield(C, 'data') && isfield(C.data, 'rt_sampen_m')
    m = C.data.rt_sampen_m;
end
if isfield(C, 'data') && isfield(C.data, 'rt_sampen_r')
    r = C.data.rt_sampen_r;
end
if isfield(C, 'data') && isfield(C.data, 'rt_sampen_min_n')
    min_n_sampen = C.data.rt_sampen_min_n;
end

for isubj = 1:nSubj
    steady_data = load(fullfile(behavior_path, steady_files(isubj).name), 'steady_behavior');
    unsteady_data = load(fullfile(behavior_path, unsteady_files(isubj).name), 'unsteady_behavior');

    steady_behavior = steady_data.steady_behavior;
    unsteady_behavior = unsteady_data.unsteady_behavior;

    % ACC on all trials
    R.steady.acc(isubj, 1) = mean(steady_behavior(:, 2) == 1);
    R.unsteady.acc(isubj, 1) = mean(unsteady_behavior(:, 2) == 1);

    % correct trials only for RT analyses
    steady_correct = steady_behavior(steady_behavior(:, 2) == 1, :);
    unsteady_correct = unsteady_behavior(unsteady_behavior(:, 2) == 1, :);

    R.steady = summarize_rt_block(R.steady, steady_correct, isubj, m, r, min_n_sampen);
    R.unsteady = summarize_rt_block(R.unsteady, unsteady_correct, isubj, m, r, min_n_sampen);
end
end

function out = summarize_rt_block(out, behavior, subj_idx, m, r, min_n_sampen)

rt = behavior(:, 3);
rt = rt(isfinite(rt));

out.all.mean(subj_idx, 1) = mean(rt, 'omitnan');
out.all.sd(subj_idx, 1) = std(rt, 'omitnan');
out.all.cv(subj_idx, 1) = out.all.sd(subj_idx, 1) / out.all.mean(subj_idx, 1);
out.all.sampen(subj_idx, 1) = safe_sampen(rt, m, r, min_n_sampen);

idx_f08 = behavior(:, 4) == 2;
idx_f12 = behavior(:, 4) == 1;

rt_f08 = behavior(idx_f08, 3);
rt_f08 = rt_f08(isfinite(rt_f08));

rt_f12 = behavior(idx_f12, 3);
rt_f12 = rt_f12(isfinite(rt_f12));

out.f08.mean(subj_idx, 1) = mean(rt_f08, 'omitnan');
out.f08.sd(subj_idx, 1) = std(rt_f08, 'omitnan');
out.f08.sampen(subj_idx, 1) = safe_sampen(rt_f08, m, r, min_n_sampen);

out.f12.mean(subj_idx, 1) = mean(rt_f12, 'omitnan');
out.f12.sd(subj_idx, 1) = std(rt_f12, 'omitnan');
out.f12.sampen(subj_idx, 1) = safe_sampen(rt_f12, m, r, min_n_sampen);

end

function v = safe_sampen(x, m, r, min_n_sampen)

x = x(:)';
x = x(isfinite(x));

if numel(x) < min_n_sampen
    v = NaN;
    return;
end

if std(x) == 0
    v = 0;
    return;
end

try
    v = sampen(x, m, r, 'chebychev');
catch
    v = NaN;
end

if ~isfinite(v)
    v = NaN;
end
end