function [id_accuracy, acc_per_pair, chance_level] = fingerprinting_universal(data)
% FINGERPRINTING_UNIVERSAL  Individual fingerprinting across sessions.
%
% Accepts either:
%   FC  input : [nROI x nROI x nSubj x nSes]  (correlation matrix per subject/session)
%   PSD input : [nROI x nSubj x nSes]          (spatial pattern per subject/session)
%
% For FC input:
%   Uses upper-triangle of the matrix as feature vector, then computes
%   correlation across all ROI pairs between database and probe subjects.
%
% For PSD input:
%   Uses the ROI vector as spatial pattern, same logic as Finn et al. 2015.
%
% For each (database_ses, probe_ses) pair with i ~= j:
%   Each probe subject is matched to the database subject with highest
%   feature correlation. Identification succeeds if the match is correct.
%
% INPUT:
%   data : [nROI x nROI x nSubj x nSes]  → FC mode
%          [nROI x nSubj x nSes]          → PSD mode
%
% OUTPUT:
%   id_accuracy  : scalar, mean identification accuracy across all session pairs
%   acc_per_pair : [nSes x nSes] accuracy for each (database, probe) pair
%                  diagonal entries are NaN (same session skipped)
%   chance_level : 1/nSubj

%% ---- Detect input mode ----
nd = ndims(data);
sz = size(data);

if nd == 4
    mode = 'FC';
    nROI  = sz(1);
    nSubj = sz(3);
    nSes  = sz(4);
    fprintf('Mode: FC  [%d x %d ROIs, %d subjects, %d sessions]\n', nROI, nROI, nSubj, nSes);
elseif nd == 3
    mode = 'PSD';
    nROI  = sz(1);
    nSubj = sz(2);
    nSes  = sz(3);
    fprintf('Mode: PSD [%d ROIs, %d subjects, %d sessions]\n', nROI, nSubj, nSes);
else
    error('Input must be 3-D [nROI x nSubj x nSes] or 4-D [nROI x nROI x nSubj x nSes].');
end

chance_level = 1 / nSubj;

%% ---- Extract feature vectors ----
% Convert data to [nFeat x nSubj x nSes] regardless of input mode

if strcmp(mode, 'FC')
    % Upper triangle indices (excluding diagonal)
    mask = triu(true(nROI), 1);
    nFeat = sum(mask(:));
    features = nan(nFeat, nSubj, nSes);
    for ises = 1:nSes
        for isubj = 1:nSubj
            fc_mat = data(:, :, isubj, ises);   % [nROI x nROI]
            features(:, isubj, ises) = fc_mat(mask);
        end
    end
else  % PSD
    nFeat = nROI;
    features = data;   % already [nROI x nSubj x nSes]
end

fprintf('Feature dimensionality: %d\n', nFeat);

%% ---- Fingerprinting across session pairs ----
acc_per_pair = nan(nSes, nSes);

for i = 1:nSes
    for j = 1:nSes
        if i == j
            continue;
        end

        db    = features(:, :, i);   % [nFeat x nSubj] database
        probe = features(:, :, j);   % [nFeat x nSubj] probe

        % Keep only subjects with complete data in both sessions
        valid = all(~isnan(db), 1) & all(~isnan(probe), 1);
        if sum(valid) < 3
            warning('Session pair (%d,%d): fewer than 3 valid subjects, skipping.', i, j);
            continue;
        end

        db_v    = db(:, valid);      % [nFeat x nValid]
        probe_v = probe(:, valid);   % [nFeat x nValid]
        nValid  = sum(valid);

        % Correlation matrix: [nValid x nValid]
        % corr_mat(p, d) = correlation between probe subject p and db subject d
        corr_mat = corr(probe_v, db_v);   % [nValid x nValid]

        % Each probe subject picks the db subject with highest correlation
        [~, matched] = max(corr_mat, [], 2);   % [nValid x 1]

        acc_per_pair(i, j) = mean(matched == (1:nValid)');
    end
end

%% ---- Summary ----
id_accuracy = mean(acc_per_pair(:), 'omitnan');

fprintf('\n=== Fingerprinting Results ===\n');
fprintf('Chance level       : %.1f%%\n', chance_level * 100);
fprintf('Mean ID accuracy   : %.1f%%\n', id_accuracy * 100);
fprintf('Ratio vs chance    : %.1fx\n',  id_accuracy / chance_level);
fprintf('\nPer-pair accuracy matrix (rows=database, cols=probe):\n');
disp(round(acc_per_pair * 100, 1));

end