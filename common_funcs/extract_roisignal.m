function extract_roisignal(mask_path, sub_path, out_root, sesName)
% EXTRACT_ROISIGNAL  Extract ROI-averaged time series and save as .mat files.
%
% Inputs:
%   mask_path  - Path to atlas NIfTI file (e.g., 'BN_Atlas_246_3mm.nii')
%   sub_path   - Root directory containing subject folders (sub-XX) or .nii files
%   out_root   - Output root directory (e.g., P.data.roisignals)
%   sesName    - Session name (e.g., 'ses-run1')
%
% Output file structure:
%   out_root/sub-01/sesName/roisignals.mat
%   out_root/sub-02/sesName/roisignals.mat
%   ...
%   Each .mat file contains:
%     roisignals_noGSR  [nROI x T]  - ROI signals without global signal regression
%     roisignals_GSR    [nROI x T]  - ROI signals after global signal regression

%% ---- Load atlas mask ----
mask_data = y_Read(mask_path);
dim = size(mask_data);
mask_vec = mask_data(:);              % Flatten to vector
brain_idx = find(mask_vec ~= 0);     % Indices of non-zero voxels
mask_roi = mask_vec(brain_idx);       % ROI labels for non-zero voxels
labels = unique(mask_roi);
nROI = length(labels);

%% ---- Find subjects ----
sub_file = dir(fullfile(sub_path, 'sub*'));
if isempty(sub_file)
    sub_file = dir(sub_path);
    sub_file(1:2) = [];
end
nSub = numel(sub_file);
fprintf('Found %d subjects, %d ROIs\n', nSub, nROI);

%% ---- Extract ROI signals subject by subject ----
for i = 1:nSub
    fprintf('Subject %d/%d: %s\n', i, nSub, sub_file(i).name);

    % Load 4D brain data
    if sub_file(i).isdir
        brain_dir = fullfile(sub_path, sub_file(i).name);
        nii_file = dir(fullfile(brain_dir, '*.nii'));
        [brain_data, ~] = y_Read(fullfile(brain_dir, nii_file(1).name));
    else
        [brain_data, ~] = y_Read(fullfile(sub_path, sub_file(i).name));
    end

    nTP = size(brain_data, 4);
    brain_2d = reshape(brain_data, [], nTP);   % [nVoxel x T]
    brain_masked = brain_2d(brain_idx, :);     % [nBrainVoxel x T]

    % Compute mean signal per ROI
    roisignals_noGSR = zeros(nROI, nTP);
    for iroi = 1:nROI
        roi_pos = (mask_roi == labels(iroi));
        roisignals_noGSR(iroi, :) = mean(brain_masked(roi_pos, :), 1);
    end

    % Global signal regression (GSR)
    gs = mean(brain_masked, 1)';               % [T x 1]
    G = [gs, ones(nTP, 1)];
    roisignals_GSR = zeros(nROI, nTP);
    for iroi = 1:nROI
        y = roisignals_noGSR(iroi, :)';
        b = G \ y;
        roisignals_GSR(iroi, :) = (y - G * b)';
    end

    % Format subject folder name
    subName = sub_file(i).name;
    if ~startsWith(subName, 'sub')
        subName = sprintf('sub-%02d', i);
    end

    % Save output
    save_dir = fullfile(out_root, subName, sesName);
    if ~exist(save_dir, 'dir'), mkdir(save_dir); end
    save(fullfile(save_dir, 'roisignals.mat'), 'roisignals_noGSR', 'roisignals_GSR');
end

fprintf('Done. Results saved to %s\n', out_root);
end