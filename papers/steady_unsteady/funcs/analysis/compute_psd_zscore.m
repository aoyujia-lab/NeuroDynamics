function [psd_out, f_sel] = compute_psd_zscore(C, res)
%COMPUTE_PSD_ZSCORE Compute normalized ROI-wise PSD from GLM residuals.
%
% Inputs
%   C   : config struct with C.psd and C.data fields
%   res : struct with res.residuals [nTime x nROI x nSubj]
%
% Outputs
%   psd_out : [nFreq x nROI x nSubj] normalized power spectrum
%   f_sel   : selected frequency vector

[nTime, nROI, nSubj] = size(res.residuals);
psd_out = [];
f_sel = [];

for isubj = 1:nSubj
    for iroi = 1:nROI
        ts = double(res.residuals(:, iroi, isubj));
        ts = ts - mean(ts, 'omitnan');

        switch lower(C.psd.method)
            case 'periodogram'
                [power, freq] = periodogram(ts, [], C.psd.nfft, C.data.FS);
            otherwise
                [power, freq] = pwelch(ts, [], [], C.psd.nfft, C.data.FS);
        end

        if isempty(f_sel)
            idx_start = find(freq >= 0.01, 1, 'first');
            idx_keep = idx_start:(numel(freq) - 1);
            f_sel = freq(idx_keep);
            psd_out = zeros(numel(f_sel), nROI, nSubj);
        end

        power_sel = power(idx_keep);
        psd_out(:, iroi, isubj) = power_sel ./ sum(power_sel);
    end
end
end
