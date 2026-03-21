function [psd_out, f_sel] = compute_psd_zscore(C, res)

for isubj = 1:size(res.residuals, 3)
    for iroi = 1:size(res.residuals, 2)
        ts = double(res.residuals(:,iroi, isubj))';
        ts = ts - mean(ts, 'omitnan');

        % --- PSD ---
        if strcmpi(C.psd.method, 'periodogram')
            [p, f] = periodogram(ts, [], C.psd.nfft, C.data.FS);
        else
            [p, f] = pwelch(ts, [], [], C.psd.nfft, C.data.FS);
        end

        % Keep frequencies from ~0.01 Hz to the second last bin
        [~, idx01] = min(abs(f - 0.01));
        p_sel = p(idx01:end-1);
        f_sel = f(idx01:end-1);

        % Store z-scored PSD (per ROI, per scan)
        psd_out(:, iroi, isubj) = p_sel./sum(p_sel);
    end
end