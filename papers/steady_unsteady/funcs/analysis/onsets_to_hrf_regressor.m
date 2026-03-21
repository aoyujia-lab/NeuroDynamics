function y = onsets_to_hrf_regressor(onsets, t_highres, width, hrf)
% onsets_to_hrf_regressor
% Generic utility:
%   onset list (sec) -> high-res pulse train -> HRF convolution -> high-res regressor
%
% Inputs
%   onsets   : vector of onset times in seconds (NaN or negative will be ignored)
%   t_highres: high-res time axis (sec), monotonically increasing
%   width    : pulse width in seconds
%   hrf      : HRF vector sampled at the same dt as t_highres (column or row)
%
% Output
%   y        : HRF-convolved time series at high-res (same size as t_highres)

    if nargin < 4
        error('Need onsets, t_highres, width, and hrf.');
    end


    hrf = hrf(:); % ensure column
    dt  = t_highres(2) - t_highres(1);
    if any(abs(diff(t_highres) - dt) > 1e-12)
        error('t_highres must be uniformly sampled.');
    end

    % Build pulse train (boxcar)
    pulse  = zeros(size(t_highres));
    onsets = onsets(:);

    for i = 1:numel(onsets)
        pt = onsets(i);
        if isnan(pt) || pt < 0
            continue;
        end
        pulse((t_highres >= pt) & (t_highres < pt + width)) = 1;
    end

    % Convolution (keep same length as t_highres)
    y = conv(pulse(:), hrf, 'full');
    y = y(1:numel(t_highres));
    y = reshape(y, size(t_highres)); % match original shape
end