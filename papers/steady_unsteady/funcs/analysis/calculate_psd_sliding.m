function [psd_out, psd_out_z, psd_out_raw, freq_out, win_center_sec, subj_dir] = calculate_psd_sliding(P, C)
%CALCULATE_PSD_SLIDING_MEMSAVE Memory-efficient sliding-window PSD.
%
% OUTPUT
%   psd_out        : [nFreq x nROI x nWin x nSubj x maxSes], normalized PSD
%   psd_out_z      : optional, default []
%   psd_out_raw    : optional, default []
%   freq_out       : [nFreq x 1]
%   win_center_sec : [nWin x 1]
%
% IMPORTANT
%   By default, only psd_out is stored to save memory.
%
% OPTIONAL PARAMETERS
%   C.psd.keep_norm : true/false, default true
%   C.psd.keep_z    : true/false, default false
%   C.psd.keep_raw  : true/false, default false
%   C.psd.precision : 'single' or 'double', default 'single'
%
% Sliding-window parameters:
%   C.psd.slide_win_sec
%   C.psd.slide_step_sec

%% ---- Defaults ----
if ~isfield(C.psd, 'slide_win_sec')
    error('Please specify C.psd.slide_win_sec.');
end

if ~isfield(C.psd, 'slide_step_sec')
    C.psd.slide_step_sec = C.psd.slide_win_sec / 2;
end

if ~isfield(C.psd, 'min_freq')
    C.psd.min_freq = 0.01;
end

if ~isfield(C.psd, 'detrend')
    C.psd.detrend = false;
end

if ~isfield(C.psd, 'keep_norm')
    C.psd.keep_norm = true;
end

if ~isfield(C.psd, 'keep_z')
    C.psd.keep_z = false;
end

if ~isfield(C.psd, 'keep_raw')
    C.psd.keep_raw = false;
end

if ~isfield(C.psd, 'precision')
    C.psd.precision = 'single';
end

win_len  = round(C.psd.slide_win_sec  * C.data.FS);
step_len = round(C.psd.slide_step_sec * C.data.FS);

if win_len < 2
    error('Sliding window is too short. Check C.psd.slide_win_sec and C.data.FS.');
end

if step_len < 1
    error('Sliding step is too short. Check C.psd.slide_step_sec and C.data.FS.');
end

%% ---- Find subjects ----
subj_dir = dir(fullfile(P.data.roisignals, 'sub*'));
subj_dir = subj_dir([subj_dir.isdir]);

if isfield(C, 'data') && isfield(C.data, 'excludesubjects') && ~isempty(C.data.excludesubjects)
    subj_dir(C.data.excludesubjects) = [];
end

nSubj = numel(subj_dir);

%% ---- Find max number of sessions and max number of windows ----
nSesEach = zeros(nSubj, 1);
maxWin = 0;

for s = 1:nSubj
    ses_dir = dir(fullfile(P.data.roisignals, subj_dir(s).name, 'ses*'));
    ses_dir = ses_dir([ses_dir.isdir]);
    nSesEach(s) = numel(ses_dir);

    for ises = 1:numel(ses_dir)
        fmat = fullfile(P.data.roisignals, subj_dir(s).name, ses_dir(ises).name, 'roisignals.mat');
        S = load(fmat);

        if isfield(S, 'roisignals')
            nTR = size(S.roisignals, 2);
        elseif isfield(S, 'roisignals_GSR')
            nTR = size(S.roisignals_GSR, 2);
        elseif isfield(S, 'roisignals_noGSR')
            nTR = size(S.roisignals_noGSR, 2);
        else
            error('No ROI signal variable found in %s', fmat);
        end

        nWin_this = numel(1:step_len:(nTR - win_len + 1));
        maxWin = max(maxWin, nWin_this);
    end
end

maxSes = max(nSesEach);

%% ---- Initialize outputs ----
psd_out        = [];
psd_out_z      = [];
psd_out_raw    = [];
freq_out       = [];
win_center_sec = [];

%% ---- Main loop ----
for isubj = 1:nSubj

    fprintf('Subject %d/%d: %s\n', isubj, nSubj, subj_dir(isubj).name);

    ses_dir = dir(fullfile(P.data.roisignals, subj_dir(isubj).name, 'ses*'));
    ses_dir = ses_dir([ses_dir.isdir]);

    for ises = 1:numel(ses_dir)

        fmat = fullfile(P.data.roisignals, subj_dir(isubj).name, ses_dir(ises).name, 'roisignals.mat');
        S = load(fmat);

        %% ---- Select ROI signals ----
        if C.data.GSR == 1
            if isfield(S, 'roisignals_GSR')
                roisignals = S.roisignals_GSR;
            else
                roisignals = S.roisignals;
            end
        else
            if isfield(S, 'roisignals_noGSR')
                roisignals = S.roisignals_noGSR;
            else
                roisignals = S.roisignals;
            end
        end

        nTR = size(roisignals, 2);

        %% ---- Define sliding windows ----
        win_start = 1:step_len:(nTR - win_len + 1);
        nWin = numel(win_start);

        if nWin < 1
            error('Sliding window is longer than the time series.');
        end

        curr_win_center_sec = ((win_start(:) - 1) + (win_len - 1) / 2) ./ C.data.FS;

        %% ---- ROI loop ----
        for iroi = 1:size(roisignals, 1)

            ts = double(roisignals(iroi, :)).';

            for iwin = 1:nWin

                idx = win_start(iwin):(win_start(iwin) + win_len - 1);
                ts_win = ts(idx);

                %% ---- PSD ----
            
                    [p, f] = periodogram(ts_win, [], C.psd.nfft, C.data.FS);
             

                %% ---- Select frequencies ----
                [~, idx_min] = min(abs(f - C.psd.min_freq));

                p_sel = p(idx_min:end-1);
                f_sel = f(idx_min:end-1);

                %% ---- Allocate only requested outputs ----
                if isempty(freq_out)

                    nFreq = numel(f_sel);
                    freq_out = f_sel;
                    win_center_sec = nan(maxWin, 1);
                    win_center_sec(1:nWin) = curr_win_center_sec;

                    fprintf('nFreq = %d, nROI = %d, maxWin = %d, nSubj = %d, maxSes = %d\n', ...
                        nFreq, C.data.nROI, maxWin, nSubj, maxSes);

                    one_array_GB = nFreq * C.data.nROI * maxWin * nSubj * maxSes * bytes_per_value(C.psd.precision) / 1024^3;
                    fprintf('Estimated memory per stored PSD array: %.2f GB\n', one_array_GB);

                    if C.psd.keep_norm
                        psd_out = nan(nFreq, C.data.nROI, maxWin, nSubj, maxSes, C.psd.precision);
                    end

                    if C.psd.keep_z
                        psd_out_z = nan(nFreq, C.data.nROI, maxWin, nSubj, maxSes, C.psd.precision);
                    end

                    if C.psd.keep_raw
                        psd_out_raw = nan(nFreq, C.data.nROI, maxWin, nSubj, maxSes, C.psd.precision);
                    end

                else

                    if numel(f_sel) ~= numel(freq_out) || any(abs(f_sel - freq_out) > 1e-12)
                        error('Frequency vector differs across scans. Check nfft, FS, or time series length.');
                    end

                    if length(curr_win_center_sec) > length(win_center_sec)
                        win_center_sec(1:length(curr_win_center_sec)) = curr_win_center_sec;
                    end

                end

                %% ---- Store only requested outputs ----
                if C.psd.keep_raw
                    psd_out_raw(:, iroi, iwin, isubj, ises) = cast(p_sel, C.psd.precision);
                end

                if C.psd.keep_z
                    psd_out_z(:, iroi, iwin, isubj, ises) = cast(zscore(p_sel), C.psd.precision);
                end

                if C.psd.keep_norm
                    psd_out(:, iroi, iwin, isubj, ises) = cast(p_sel ./ sum(p_sel), C.psd.precision);
                end

            end
        end
    end
end

end


function b = bytes_per_value(precision)
    switch lower(precision)
        case 'single'
            b = 4;
        case 'double'
            b = 8;
        otherwise
            error('precision must be ''single'' or ''double''.');
    end
end