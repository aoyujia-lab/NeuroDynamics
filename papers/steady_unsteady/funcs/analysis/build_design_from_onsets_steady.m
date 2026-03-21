function X = build_design_from_onsets_steady(L_onsets, R_onsets, C)
% build_design_from_onsets_articleX
% Article-specific design matrix builder for this project.
%
% Inputs
%   L_onsets, R_onsets : onset times in seconds (can include NaN)
%   P : struct with minimal fields:
%       C.glm.totalDuration   (sec) e.g., 600
%       C.data.TR              (sec) e.g., 2
%       C.glm.highResFactor   (Hz multiplier) e.g., 100 -> dt=0.01 sec
%       C.glm.pulseWidth      (sec) e.g., 0.1
%       C.glm.dropTR          (integer) e.g., 10
%       (optional) C.data.nTR : number of TRs kept AFTER drop (default: full length after drop)
%
% Output
%   X : [regL regR intercept] in TR space, AFTER dropping first P.dropTR TRs

    % ---- High-res time axis (kept outside "generic" pulse->HRF tool) ----
    dt_highres = 1 / C.glm.highResFactor;
    t_highres  = 0:dt_highres:C.glm.totalDuration;  % includes endpoint

    % ---- HRF (high-res) ----
    % Requires SPM on path
    hrf = spm_hrf(dt_highres);
    hrf = hrf(:);
    if max(hrf) ~= 0
        hrf = hrf ./ max(hrf);
    end

    % ---- High-res HRF-convolved regressors (generic tool) ----
    convL = onsets_to_hrf_regressor(L_onsets, t_highres, C.glm.pulseWidth, hrf);
    convR = onsets_to_hrf_regressor(R_onsets, t_highres, C.glm.pulseWidth, hrf);

    % ---- Downsample to TR space ----
    downsampleFactor = C.data.TR / dt_highres;  % e.g., 2/0.01 = 200
    if abs(downsampleFactor - round(downsampleFactor)) > 1e-9
        error('TR / dt_highres must be integer. Got %.6f', downsampleFactor);
    end
    downsampleFactor = round(downsampleFactor);

    sL = convL(1:downsampleFactor:end);
    sR = convR(1:downsampleFactor:end);
    sL = sL(:); sR = sR(:);

    % ---- Determine TR length and drop initial TRs ----
    % Total TR count implied by totalDuration and TR:
    % Note: because t_highres includes endpoint, the downsampled length is typically floor(totalDuration/TR)+1
    % For fMRI runs, you usually want exactly totalDuration/TR points (e.g., 600/2=300).
    % We'll enforce that behavior robustly:
    nTR_expected = round(C.glm.totalDuration / C.data.TR);
    if nTR_expected <= 0
        error('Invalid totalDuration/TR. totalDuration=%.3f, TR=%.3f', C.glm.totalDuration, C.data.TR);
    end

    if numel(sL) < nTR_expected || numel(sR) < nTR_expected
        error('Downsampled regressors shorter than expected TR count (%d). Check t_highres construction.', nTR_expected);
    end

    sL = sL(1:nTR_expected);
    sR = sR(1:nTR_expected);

    % Drop first P.dropTR TRs (article-specific choice)
    if C.glm.dropTR < 0 || C.glm.dropTR >= nTR_expected
        error('dropTR must be in [0, nTR_expected-1]. Got %d (nTR_expected=%d)', C.glm.dropTR, nTR_expected);
    end
    sL = sL(C.glm.dropTR+1:end);
    sR = sR(C.glm.dropTR+1:end);


    X = [sL, sR, ones(C.data.nTR, 1)];
end


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
    if isempty(t_highres) || numel(t_highres) < 2
        error('t_highres must have at least 2 points.');
    end
    if width <= 0
        error('width must be positive.');
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
