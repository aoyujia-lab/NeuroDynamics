function [psd_out, psd_out_z, psd_out_raw, freq_out, subj_dir] = calculate_psd(P, C)
%CALCULATE_PSD Compute PSD for ROI time series across subjects and sessions.
%
% OUTPUT
%   psd_out  : [nFreq x nROI x nSubj x maxSes] PSD (z-scored within each ROI/scan)
%   freq_out : [nFreq x 1] frequency vector (starts near 0.01 Hz)

% REQUIRED INPUT STRUCTURE

%   P.data.roisignals : char/string

%   C.data.nTR : scalar
%   C.data.FS : scalar
%   C.data.nROI : scalar
%   C.data.excludesubjects : vector (optional)
%   C.psd.method : char/string
%   C.psd.nfft : scalar

%   If C.psd.method is not 'periodogram', the following are also required:
%   C.psd.win_sec : scalar or vector
%   C.psd.overlap : scalar


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

% Allocate after first PSD so we know frequency length
psd_out  = [];
freq_out = [];

%% ---- Main loop ----
for isubj = 1:nSubj
    fprintf('Subject %d/%d: %s\n', isubj, nSubj, subj_dir(isubj).name);

    ses_dir = dir(fullfile(P.data.roisignals, subj_dir(isubj).name, 'ses*'));
    ses_dir = ses_dir([ses_dir.isdir]);

    for ises = 1:numel(ses_dir)
        fmat = fullfile(P.data.roisignals, subj_dir(isubj).name, ses_dir(ises).name, 'roisignals.mat');
        S = load(fmat);               % expects S.roisignals

        if C.data.GSR == 1
            if isfield(S, 'roisignals_GSR')
                roisignals = S.roisignals_GSR;
            else
                roisignals = S.roisignals;
            end
        else
            roisignals = S.roisignals_noGSR;
        end


        for iroi = 1:size(roisignals, 1)
            ts = double(roisignals(iroi, :)).';
            % ts = zscore(ts);

            % --- PSD ---
            if strcmpi(C.psd.method, 'periodogram')
                [p, f] = periodogram(ts, [], C.psd.nfft, C.data.FS);
            else
                [p, f] = pwelch(ts, C.psd.win_sec, C.psd.overlap, C.psd.nfft, C.data.FS);
            end

            % Keep frequencies from ~0.01 Hz to the second last bin
            [~, idx01] = min(abs(f - 0.01));
            p_sel = p(idx01:end-1);
            f_sel = f(idx01:end-1);

            % Allocate outputs once we know nFreq
            if isempty(psd_out)
                nFreq   = numel(f_sel);
                psd_out = nan(nFreq, C.data.nROI, nSubj, maxSes, 'double');
                freq_out = f_sel;
            end

            % Store z-scored PSD (per ROI, per scan)
            psd_out_z(:, iroi, isubj, ises) = zscore(p_sel);

            psd_out(:, iroi, isubj, ises) = p_sel./sum(p_sel);

            psd_out_raw(:, iroi, isubj, ises) = p_sel;
        end
    end
end

end
