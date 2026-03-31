function [r_mat, p_mat] = calculate_corr_filtered(P, C)
%CALCULATE_CORR_FILTERED Compute ROI-ROI correlation after filtering,
% excluding ROIs in C.data.excluderoi.
%
% INPUT
%   P : struct with P.data.roisignals pointing to root folder
%   C : config struct, expects:
%       C.data.nROI
%       C.data.sesnum
%       C.data.excludesubjects (optional)
%       C.data.excluderoi (optional) e.g., [120 300]
%       C.preproc.band_cutoff
%
% OUTPUT
%   r_mat    : [nROI_keep x nROI_keep x nSubj x maxSes]
%   p_mat    : [nROI_keep x nROI_keep x nSubj x maxSes]
%   roi_keep : indices of kept ROIs (in original ROI indexing)

%% ---- Find subjects (and remove excluded ones) ----
subj_dir = dir(fullfile(P.data.roisignals, 'sub*'));
subj_dir = subj_dir([subj_dir.isdir]);

if isfield(C, 'data') && isfield(C.data, 'excludesubjects') && ~isempty(C.data.excludesubjects)
    subj_dir(C.data.excludesubjects) = [];
end

nSubj = numel(subj_dir);
maxSes = C.data.sesnum;

%% ---- ROI include/exclude ----
nROI_all = C.data.nROI;

roi_exclude = [];
if isfield(C, 'data') && isfield(C.data, 'excluderoi') && ~isempty(C.data.excluderoi)
    roi_exclude = unique(C.data.excluderoi(:)');
end

roi_keep = setdiff(1:nROI_all, roi_exclude, 'stable');
nROI_keep = numel(roi_keep);

%% ---- Preallocate outputs ----
r_mat = nan(nROI_keep, nROI_keep, nSubj, maxSes, 'double');
p_mat = nan(nROI_keep, nROI_keep, nSubj, maxSes, 'double');

%% ---- Main loop ----
for isubj = 1:nSubj
    fprintf('Subject %d/%d: %s\n', isubj, nSubj, subj_dir(isubj).name);

    ses_dir = dir(fullfile(P.data.roisignals, subj_dir(isubj).name, 'ses*'));
    ses_dir = ses_dir([ses_dir.isdir]);

    for ises = 1:numel(ses_dir)
        fmat = fullfile(P.data.roisignals, subj_dir(isubj).name, ses_dir(ises).name, 'roisignals.mat');
        S = load(fmat);                 % expects S.roisignals
        if C.data.GSR == 1
            roisignals = S.roisignals_GSR;    % [ROI x T]
        else
            roisignals = S.roisignals_noGSR;    % [ROI x T]
        end

        % ---- keep only selected ROIs ----
        roisignals = roisignals(roi_keep, :);  % [nROI_keep x T]

        % ---- Filter ----
        % filter input: [T x ROI]
        roisignals_filt = y_IdealFilter(roisignals', 2, C.preproc.band_cutoff); % [T x nROI_keep]

        % ---- Correlation ----
        [R, Pval] = corr(roisignals_filt, ...
            'Type', 'Pearson', ...
            'Rows', 'pairwise');

        r_mat(:, :, isubj, ises) = R;
        p_mat(:, :, isubj, ises) = Pval;
    end
end

end
