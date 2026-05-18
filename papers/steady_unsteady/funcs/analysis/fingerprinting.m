function [id_accuracy, acc_per_pair] = fingerprinting(psd_all)
% Individual fingerprinting across runs based on spatial PSD pattern
%
% For each frequency, tests whether subjects can be identified from their
% spatial PSD pattern (across ROIs) across different runs. Each subject's
% pattern in run_j is matched against all subjects' patterns in run_i via
% spatial correlation; identification succeeds if the highest correlation
% is with the correct subject.
%
% INPUT:
%   psd_all : [nFreq × nROI × nSubj × nRun]
%
% OUTPUT:
%   id_accuracy  : [nFreq × 1] mean identification accuracy across run pairs
%   acc_per_pair : [nFreq × nRun × nRun] accuracy for each (database, probe) run pair

[nFreq, ~, nSubj, nRun] = size(psd_all);

acc_per_pair = nan(nFreq, nRun, nRun);

for f = 1:nFreq
    % [nSubj × nROI] for each run
    patterns = squeeze(psd_all(f, :, :, :));   % [nROI × nSubj × nRun]
    
    for i = 1:nRun
        for j = 1:nRun
            if i == j
                continue;
            end
            
            db    = patterns(:, :, i);   % [nROI × nSubj] database
            probe = patterns(:, :, j);   % [nROI × nSubj] probe
            
            % only keep subjects with valid data in both runs
            valid = all(~isnan(db), 1) & all(~isnan(probe), 1);
            if sum(valid) < 3
                continue;
            end
            db_v    = db(:, valid);
            probe_v = probe(:, valid);
            nS      = sum(valid);
            
            % correlation matrix: rows = probe subjects, cols = db subjects
            corr_mat = corr(probe_v, db_v);   % [nS × nS]
            
            % for each probe, find best match in database
            [~, matched] = max(corr_mat, [], 2);
            acc_per_pair(f, i, j) = mean(matched == (1:nS)');
        end
    end
    
    if mod(f, 100) == 0
        fprintf('freq %d/%d done\n', f, nFreq);
    end
end

% average across all off-diagonal (i, j) pairs
id_accuracy = mean(acc_per_pair, [2 3], 'omitnan');   % [nFreq × 1]

end