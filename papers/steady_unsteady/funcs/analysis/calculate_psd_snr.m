function [psd_out, psd_out_snr, freq_out] = calculate_psd_snr(P, C)
%CALCULATE_PSD_SNR  Compute PSD and SNR for ROI time series.
%
% SNR definition (Norcia et al., 2015, Appendix 2):
%   SNR(f) = amplitude(f) / mean(amplitude of neighboring bins)
%   where neighboring bins = [f - n*df, ..., f - df, f + df, ..., f + n*df]
%   excluding the target bin itself. This interpolates the background
%   spectral slope, correcting for 1/f structure.
%
% OUTPUT
%   psd_out     : [nFreq x nROI x nSubj x maxSes]  raw PSD (power)
%   psd_out_snr : [nFreq x nROI x nSubj x maxSes]  SNR (amplitude ratio)
%   freq_out    : [nFreq x 1] frequency vector (starts near 0.01 Hz)
%
% ADDITIONAL FIELD REQUIRED IN C:
%   C.psd.snr_neighbors : number of neighboring bins on each side (e.g. 10)
%                         These bins straddle the target bin symmetrically.

%% ---- Parameter ----
n_neighbors = C.psd.snr_neighbors;   % e.g. 10 bins on each side

%% ---- Find subjects ----
subj_dir = dir(fullfile(P.data.roisignals, 'sub*'));
subj_dir = subj_dir([subj_dir.isdir]);
if isfield(C, 'data') && isfield(C.data, 'excludesubjects') && ~isempty(C.data.excludesubjects)
    subj_dir(C.data.excludesubjects) = [];
end
nSubj = numel(subj_dir);

%% ---- Find max sessions ----
nSesEach = zeros(nSubj, 1);
for s = 1:nSubj
    ses_dir = dir(fullfile(P.data.roisignals, subj_dir(s).name, 'ses*'));
    ses_dir = ses_dir([ses_dir.isdir]);
    nSesEach(s) = numel(ses_dir);
end
maxSes = max(nSesEach);

psd_out     = [];
psd_out_snr = [];
freq_out    = [];

%% ---- Main loop ----
for isubj = 1:nSubj
    fprintf('Subject %d/%d: %s\n', isubj, nSubj, subj_dir(isubj).name);
    ses_dir = dir(fullfile(P.data.roisignals, subj_dir(isubj).name, 'ses*'));
    ses_dir = ses_dir([ses_dir.isdir]);

    for ises = 1:numel(ses_dir)
        fmat = fullfile(P.data.roisignals, subj_dir(isubj).name, ...
                        ses_dir(ises).name, 'roisignals.mat');
        S = load(fmat);

        if C.data.GSR == 1
            roisignals = S.roisignals_GSR;
        else
            roisignals = S.roisignals_noGSR;
        end

        for iroi = 1:size(roisignals, 1)
            ts = double(roisignals(iroi, :)).';

            %% --- Compute PSD ---
            if strcmpi(C.psd.method, 'periodogram')
                [p, f] = periodogram(ts, [], C.psd.nfft, C.data.FS);
            else
                [p, f] = pwelch(ts, C.psd.win_sec, C.psd.overlap, ...
                                C.psd.nfft, C.data.FS);
            end

            %% --- Frequency selection ---
            [~, idx01] = min(abs(f - 0.01));
            p_sel = p(idx01:end-1);
            f_sel = f(idx01:end-1);
            nFreq_sel = numel(f_sel);

            %% --- Allocate on first pass ---
            if isempty(psd_out)
                nFreq       = nFreq_sel;
                psd_out     = nan(nFreq, C.data.nROI, nSubj, maxSes);
                psd_out_snr = nan(nFreq, C.data.nROI, nSubj, maxSes);
                freq_out    = f_sel;
            end

            %% --- Compute amplitude spectrum ---
            % Power -> amplitude (sqrt) for SNR computation.
            % Norcia et al. (2015) App.2: SNR = A(f) / mean(A(neighbors))
            amp_sel = sqrt(p_sel);   % amplitude spectrum

            %% --- Compute SNR per bin ---
            snr_sel = compute_snr(amp_sel, n_neighbors);

            %% --- Store ---
            psd_out    (:, iroi, isubj, ises) = p_sel;
            psd_out_snr(:, iroi, isubj, ises) = snr_sel;
        end
    end
end
end % end main function


%% =========================================================
function snr = compute_snr(amp, n_neighbors)
%COMPUTE_SNR  SNR via neighboring-bin interpolation (Norcia 2015, App.2).
%
%   For each bin i, noise is estimated as the mean amplitude of the
%   n_neighbors bins immediately below and n_neighbors bins immediately
%   above bin i (2*n_neighbors bins total).  Bins at the edges where
%   fewer neighbors are available are set to NaN.
%
%   SNR(i) = amp(i) / mean( amp([i-n : i-1, i+1 : i+n]) )
%
%   This symmetric window straddles the target bin and therefore
%   interpolates the local spectral background, correcting for the
%   slope of the 1/f spectrum.

N   = numel(amp);
snr = nan(N, 1);

for i = 1:N
    lo = i - n_neighbors;   % first lower neighbor index
    hi = i + n_neighbors;   % last upper neighbor index

    % Skip bins where the full symmetric window is not available
    if lo < 1 || hi > N
        continue
    end

    % Neighbor indices: exclude the target bin itself
    neighbor_idx = [lo : i-1,  i+1 : hi];
    noise_amp    = mean(amp(neighbor_idx));

    if noise_amp > 0
        snr(i) = amp(i) / noise_amp;
    end
end
end