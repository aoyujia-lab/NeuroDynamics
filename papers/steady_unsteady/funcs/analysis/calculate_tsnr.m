function [tsnr_out, tsnr_mean] = calculate_tsnr(P, C)
%CALCULATE_TSNR Compute tSNR (mean/SD) for ROI time series across subjects and sessions.
%
% OUTPUT
%   tsnr_out  : [nROI x nSubj x maxSes] tSNR values (mean/SD per ROI per scan)
%   tsnr_mean : [nROI x 1] tSNR averaged across subjects and sessions
%
% REQUIRED INPUT STRUCTURE
%   P.data.roisignals : char/string
%   C.data.nROI : scalar
%   C.data.GSR : scalar (0 or 1)
%   C.data.excludesubjects : vector (optional)

%% ---- Find subjects (and remove excluded ones) ----
subj_dir = dir(fullfile(P.data.roisignals, 'sub*'));
subj_dir = subj_dir([subj_dir.isdir]);
if isfield(C, 'data') && isfield(C.data, 'excludesubjects') && ~isempty(C.data.excludesubjects)
    subj_dir(C.data.excludesubjects) = [];
end
nSubj = numel(subj_dir);

%% ---- Find max number of sessions (for preallocation) ----
nSesEach = zeros(nSubj, 1);
for s = 1:nSubj
    ses_dir = dir(fullfile(P.data.roisignals, subj_dir(s).name, 'ses*'));
    ses_dir = ses_dir([ses_dir.isdir]);
    nSesEach(s) = numel(ses_dir);
end
maxSes = max(nSesEach);

% Preallocate
tsnr_out = nan(C.data.nROI, nSubj, maxSes, 'double');

%% ---- Main loop ----
for isubj = 1:nSubj
    fprintf('Subject %d/%d: %s\n', isubj, nSubj, subj_dir(isubj).name);
    ses_dir = dir(fullfile(P.data.roisignals, subj_dir(isubj).name, 'ses*'));
    ses_dir = ses_dir([ses_dir.isdir]);

    for ises = 1:numel(ses_dir)
        fmat = fullfile(P.data.roisignals, subj_dir(isubj).name, ses_dir(ises).name, 'roisignals.mat');
        S = load(fmat);               % expects S.roisignals

        if C.data.GSR == 1
            roisignals = S.roisignals_GSR;    % [ROI x T]
        else
            roisignals = S.roisignals_noGSR;  % [ROI x T]
        end

        for iroi = 1:size(roisignals, 1)
            ts = double(roisignals(iroi, :)).';

            % --- tSNR = mean / SD ---
            ts_mean = mean(ts);
            ts_sd   = std(ts);

            if ts_sd > 0
                tsnr_out(iroi, isubj, ises) = ts_mean / ts_sd;
            else
                tsnr_out(iroi, isubj, ises) = NaN;  % avoid division by zero
            end
        end
    end
end

%% ---- Average tSNR across subjects and sessions (ignoring NaNs) ----
tsnr_mean = squeeze(mean(tsnr_out, [2 3], 'omitnan'));  % [nROI x 1]

end