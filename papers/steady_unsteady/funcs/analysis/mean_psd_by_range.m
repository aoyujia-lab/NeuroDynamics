function out = mean_psd_by_range(psd, range, range_label)
% mean_psd_by_range
% Compute band-averaged PSD for each frequency range (one struct per range).
%
% INPUT
%   psd         : [nFreq x nROI x nSubj x nSes]
%   range       : cell array of frequency indices (e.g., {1:42, 75:77, ...})
%   range_label : (optional) cell array of labels, same length as range
%
% OUTPUT
%   out(ir).label        : label of frequency range
%   out(ir).idx          : frequency indices
%   out(ir).power_roi    : [nROI x nSubj x nSes]
%   out(ir).power_brain  : [nSubj x nSes]

    if nargin < 3 || isempty(range_label)
        range_label = arrayfun(@(k) sprintf('range_%d', k), ...
                               1:numel(range), 'UniformOutput', false);
    end

    nRange = numel(range);
    assert(numel(range_label) == nRange, ...
        'range_label must have same length as range.');

    [nFreq, nROI, nSubj, nSes] = size(psd);

    % precompute brain-mean PSD
    psd_brain = squeeze(mean(psd, 2, 'omitnan'));  % [nFreq x nSubj x nSes]

    out = struct([]);

    for ir = 1:nRange
        idx = range{ir};

        % safety check
        assert(all(idx >= 1 & idx <= nFreq), ...
            'Range %d has invalid frequency indices.', ir);

        out(ir).label = range_label{ir};
        out(ir).idx   = idx;

        % ---- ROI-level ----
        % mean over frequency only
        out(ir).power_roi = squeeze(mean(psd(idx,:,:,:), 1, 'omitnan'));
        % [nROI x nSubj x nSes]

        % ---- Brain-average ----
        out(ir).power_brain = squeeze(mean(psd_brain(idx,:,:), 1, 'omitnan'));
        % [nSubj x nSes]
    end
end
