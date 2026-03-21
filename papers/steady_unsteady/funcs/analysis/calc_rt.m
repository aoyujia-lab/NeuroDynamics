function R = calc_rt(P, C)
%CALC_RT Compute RT summary statistics for steady and unsteady tasks.

behavior_path = P.data.behav;
steady_files = dir(fullfile(behavior_path, '*_steady*'));
unsteady_files = dir(fullfile(behavior_path, '*_unsteady*'));

steady_files(C.data.excludesubjects) = [];
unsteady_files(C.data.excludesubjects) = [];

nSubj = numel(steady_files);

for isubj = 1:nSubj
    steady_data = load(fullfile(behavior_path, steady_files(isubj).name), 'steady_behavior');
    unsteady_data = load(fullfile(behavior_path, unsteady_files(isubj).name), 'unsteady_behavior');

    steady_behavior = steady_data.steady_behavior;
    unsteady_behavior = unsteady_data.unsteady_behavior;

    R.steady.acc(isubj) = mean(steady_behavior(:, 2) == 1);
    R.unsteady.acc(isubj) = mean(unsteady_behavior(:, 2) == 1);

    steady_correct = steady_behavior(steady_behavior(:, 2) == 1, :);
    unsteady_correct = unsteady_behavior(unsteady_behavior(:, 2) == 1, :);

    R.steady = summarize_rt_block(R.steady, steady_correct, isubj);
    R.unsteady = summarize_rt_block(R.unsteady, unsteady_correct, isubj);
end
end

function out = summarize_rt_block(out, behavior, subj_idx)
out.all.mean(subj_idx) = mean(behavior(:, 3));
out.all.sd(subj_idx) = std(behavior(:, 3));

idx_f08 = behavior(:, 4) == 2;
idx_f12 = behavior(:, 4) == 1;

out.f08.mean(subj_idx) = mean(behavior(idx_f08, 3));
out.f08.sd(subj_idx) = std(behavior(idx_f08, 3));
out.f12.mean(subj_idx) = mean(behavior(idx_f12, 3));
out.f12.sd(subj_idx) = std(behavior(idx_f12, 3));
end
