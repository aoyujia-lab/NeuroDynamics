function stats = compare_fingerprinting_features(dataA, dataB, nPerm)
% COMPARE_FINGERPRINTING_FEATURES
% Compare fingerprinting ID accuracy between two feature sets.
%
% This function performs a subject-level paired permutation test.
%
% INPUT:
%   dataA : feature A
%           FC  mode: [nROI x nROI x nSubj x nSes]
%           PSD mode: [nROI x nSubj x nSes]
%
%   dataB : feature B
%           Same subject/session structure as dataA.
%           Feature dimensionality can differ.
%
%   nPerm : number of sign-flip permutations, e.g. 5000 or 10000
%
% OUTPUT:
%   stats : structure containing mean accuracies, difference, p-values,
%           confidence interval, and subject-level accuracies.

if nargin < 3 || isempty(nPerm)
    nPerm = 10000;
end

%% Extract feature matrices
featA = extract_fingerprint_features(dataA);
featB = extract_fingerprint_features(dataB);

[nFeatA, nSubjA, nSesA] = size(featA);
[nFeatB, nSubjB, nSesB] = size(featB);

if nSubjA ~= nSubjB || nSesA ~= nSesB
    error('dataA and dataB must have the same number of subjects and sessions.');
end

nSubj = nSubjA;
nSes  = nSesA;

fprintf('Feature A dimensionality: %d\n', nFeatA);
fprintf('Feature B dimensionality: %d\n', nFeatB);
fprintf('Subjects: %d, Sessions: %d\n', nSubj, nSes);

%% Compute subject-level correctness
correctA = nan(nSubj, nSes, nSes);
correctB = nan(nSubj, nSes, nSes);

acc_pair_A = nan(nSes, nSes);
acc_pair_B = nan(nSes, nSes);

for i = 1:nSes
    for j = 1:nSes

        if i == j
            continue;
        end

        dbA    = featA(:, :, i);
        probeA = featA(:, :, j);

        dbB    = featB(:, :, i);
        probeB = featB(:, :, j);

        % Use only subjects valid for both features.
        validA = all(~isnan(dbA), 1) & all(~isnan(probeA), 1);
        validB = all(~isnan(dbB), 1) & all(~isnan(probeB), 1);

        valid = validA & validB;

        if sum(valid) < 3
            warning('Session pair (%d,%d): fewer than 3 valid subjects, skipping.', i, j);
            continue;
        end

        idxValid = find(valid);

        successA = compute_pair_success(dbA(:, valid), probeA(:, valid));
        successB = compute_pair_success(dbB(:, valid), probeB(:, valid));

        correctA(idxValid, i, j) = successA;
        correctB(idxValid, i, j) = successB;

        acc_pair_A(i, j) = mean(successA, 'omitnan');
        acc_pair_B(i, j) = mean(successB, 'omitnan');
    end
end

%% Subject-level accuracy
subj_acc_A = squeeze(mean(correctA, [2 3], 'omitnan'));
subj_acc_B = squeeze(mean(correctB, [2 3], 'omitnan'));

diff_subj = subj_acc_A - subj_acc_B;

validSubj = ~isnan(diff_subj);
diff_subj = diff_subj(validSubj);
subj_acc_A_valid = subj_acc_A(validSubj);
subj_acc_B_valid = subj_acc_B(validSubj);

nValidSubj = numel(diff_subj);

if nValidSubj < 3
    error('Fewer than 3 valid subjects for comparison.');
end

%% Observed statistics
mean_acc_A = mean(subj_acc_A_valid, 'omitnan');
mean_acc_B = mean(subj_acc_B_valid, 'omitnan');
delta_acc  = mean(diff_subj, 'omitnan');

sd_diff = std(diff_subj, 'omitnan');

if sd_diff > 0
    cohen_dz = delta_acc / sd_diff;
else
    cohen_dz = NaN;
end

%% Paired sign-flip permutation test
perm_delta = nan(nPerm, 1);

for iperm = 1:nPerm
    signs = randi([0 1], nValidSubj, 1) * 2 - 1;
    perm_delta(iperm) = mean(diff_subj .* signs, 'omitnan');
end

p_two_sided = (sum(abs(perm_delta) >= abs(delta_acc)) + 1) / (nPerm + 1);
p_A_greater = (sum(perm_delta >= delta_acc) + 1) / (nPerm + 1);
p_B_greater = (sum(perm_delta <= delta_acc) + 1) / (nPerm + 1);

%% Bootstrap confidence interval over subjects
nBoot = nPerm;
boot_delta = nan(nBoot, 1);

for iboot = 1:nBoot
    bootIdx = randi(nValidSubj, nValidSubj, 1);
    boot_delta(iboot) = mean(diff_subj(bootIdx), 'omitnan');
end

ci_delta = prctile(boot_delta, [2.5 97.5]);

%% Store results
stats = struct();

stats.mean_acc_A = mean_acc_A;
stats.mean_acc_B = mean_acc_B;
stats.delta_acc  = delta_acc;

stats.mean_acc_A_percent = mean_acc_A * 100;
stats.mean_acc_B_percent = mean_acc_B * 100;
stats.delta_acc_percent  = delta_acc  * 100;

stats.p_two_sided = p_two_sided;
stats.p_A_greater = p_A_greater;
stats.p_B_greater = p_B_greater;

stats.ci_delta = ci_delta;
stats.ci_delta_percent = ci_delta * 100;

stats.cohen_dz = cohen_dz;

stats.nValidSubj = nValidSubj;
stats.nPerm = nPerm;

stats.subj_acc_A = subj_acc_A;
stats.subj_acc_B = subj_acc_B;
stats.diff_subj = subj_acc_A - subj_acc_B;

stats.correctA = correctA;
stats.correctB = correctB;

stats.acc_pair_A = acc_pair_A;
stats.acc_pair_B = acc_pair_B;

%% Print summary
fprintf('\n=== Fingerprinting feature comparison ===\n');
fprintf('Feature A mean accuracy : %.2f%%\n', mean_acc_A * 100);
fprintf('Feature B mean accuracy : %.2f%%\n', mean_acc_B * 100);
fprintf('A - B difference        : %.2f percentage points\n', delta_acc * 100);
fprintf('95%% bootstrap CI        : [%.2f, %.2f] percentage points\n', ...
    ci_delta(1) * 100, ci_delta(2) * 100);
fprintf('Paired permutation p    : %.4f, two-sided\n', p_two_sided);
fprintf('One-sided p, A > B      : %.4f\n', p_A_greater);
fprintf('One-sided p, B > A      : %.4f\n', p_B_greater);
fprintf('Cohen dz                : %.3f\n', cohen_dz);
fprintf('Valid subjects          : %d\n', nValidSubj);

end


%% Helper function: extract features
function features = extract_fingerprint_features(data)
% Convert FC or PSD input into [nFeat x nSubj x nSes].

nd = ndims(data);
sz = size(data);

if nd == 4
    % FC mode: [nROI x nROI x nSubj x nSes]
    nROI  = sz(1);
    nSubj = sz(3);
    nSes  = sz(4);

    if sz(1) ~= sz(2)
        error('For FC input, the first two dimensions must be nROI x nROI.');
    end

    mask = triu(true(nROI), 1);
    nFeat = sum(mask(:));

    features = nan(nFeat, nSubj, nSes);

    for ises = 1:nSes
        for isubj = 1:nSubj
            fc_mat = data(:, :, isubj, ises);
            features(:, isubj, ises) = fc_mat(mask);
        end
    end

elseif nd == 3
    % PSD mode: [nROI x nSubj x nSes]
    features = data;

else
    error('Input must be 3-D [nROI x nSubj x nSes] or 4-D [nROI x nROI x nSubj x nSes].');
end

end


%% Helper function: compute correctness for one session pair
function success = compute_pair_success(db, probe)
% db    : [nFeat x nSubj]
% probe : [nFeat x nSubj]
%
% success : [nSubj x 1], 1 = correct, 0 = incorrect, NaN = undefined

nSubj = size(db, 2);

corr_mat = corr(probe, db);  % [probe subject x database subject]

success = nan(nSubj, 1);

finiteRow = any(isfinite(corr_mat), 2);

corr_tmp = corr_mat;
corr_tmp(~isfinite(corr_tmp)) = -Inf;

[~, matched] = max(corr_tmp, [], 2);

success(finiteRow) = matched(finiteRow) == (1:nSubj)';

end