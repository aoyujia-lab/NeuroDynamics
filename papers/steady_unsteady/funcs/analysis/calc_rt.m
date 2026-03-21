function R = calc_rt(P,C)
% CAL_RT  Compute RT statistics and accuracy for steady / unsteady tasks.
%
% OUTPUT (struct R)
%   R.steady.acc
%   R.unsteady.acc
%
%   R.steady.all.mean
%   R.steady.all.sd
%   R.steady.f08.mean
%   R.steady.f08.sd
%   R.steady.f12.mean
%   R.steady.f12.sd
%
%   R.unsteady.all.mean
%   R.unsteady.all.sd
%   R.unsteady.f08.mean
%   R.unsteady.f08.sd
%   R.unsteady.f12.mean
%   R.unsteady.f12.sd

%% ------------------------------------------------------------------------
% Locate files
behavior_path = P.data.behav;
steady_files   = dir(fullfile(behavior_path, '*_steady*'));
unsteady_files = dir(fullfile(behavior_path, '*_unsteady*'));

steady_files(C.data.excludesubjects) = [];
unsteady_files(C.data.excludesubjects) = [];

nSubj = numel(steady_files);

%% ------------------------------------------------------------------------
% Loop over subjects
for isubj = 1:nSubj

    % Load data
    load(fullfile(behavior_path, steady_files(isubj).name),   'steady_behavior');
    load(fullfile(behavior_path, unsteady_files(isubj).name), 'unsteady_behavior');

    %% ---------------- Accuracy (before removing errors) ------------------
    R.steady.acc(isubj)   = mean(steady_behavior(:,2)   == 1);
    R.unsteady.acc(isubj) = mean(unsteady_behavior(:,2) == 1);

    %% ---------------- Remove incorrect trials ----------------------------
    steady_behavior   = steady_behavior(steady_behavior(:,2)   == 1, :);
    unsteady_behavior = unsteady_behavior(unsteady_behavior(:,2) == 1, :);

    %% ---------------- All trials -----------------------------------------
    R.steady.all.mean(isubj)   = mean(steady_behavior(:,3));
    R.steady.all.sd(isubj)     = std(steady_behavior(:,3));

    R.unsteady.all.mean(isubj) = mean(unsteady_behavior(:,3));
    R.unsteady.all.sd(isubj)   = std(unsteady_behavior(:,3));

    %% ---------------- 0.8 Hz (label == 2) ---------------------------------
    idx08_stea   = steady_behavior(:,4)   == 2;
    idx08_unstea = unsteady_behavior(:,4) == 2;

    R.steady.f08.mean(isubj)   = mean(steady_behavior(idx08_stea,   3));
    R.steady.f08.sd(isubj)     = std(steady_behavior(idx08_stea,   3));

    R.unsteady.f08.mean(isubj) = mean(unsteady_behavior(idx08_unstea, 3));
    R.unsteady.f08.sd(isubj)   = std(unsteady_behavior(idx08_unstea, 3));

    %% ---------------- 1.2 Hz (label == 1) ---------------------------------
    idx12_stea   = steady_behavior(:,4)   == 1;
    idx12_unstea = unsteady_behavior(:,4) == 1;

    R.steady.f12.mean(isubj)   = mean(steady_behavior(idx12_stea,   3));
    R.steady.f12.sd(isubj)     = std(steady_behavior(idx12_stea,   3));

    R.unsteady.f12.mean(isubj) = mean(unsteady_behavior(idx12_unstea, 3));
    R.unsteady.f12.sd(isubj)   = std(unsteady_behavior(idx12_unstea, 3));

end
end
