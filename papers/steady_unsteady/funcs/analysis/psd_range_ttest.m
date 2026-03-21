function stats = psd_range_ttest(out)
%PSD_RANGE_TTEST
% Paired t-tests between sessions on band-averaged PSD (per range).
%
% INPUT
%   out : struct array from mean_psd_by_range()
%         out(ir).label       (optional)
%         out(ir).idx         (optional)
%         out(ir).power_roi   [nROI x nSubj x nSes]  or [nSubj x nSes] if nROI=1
%         out(ir).power_brain [nSubj x nSes]
%
% OUTPUT (stats struct array)
%   stats(ir).label
%   stats(ir).idx
%   stats(ir).brain.band_psd
%   stats(ir).brain.pair(ip).p_raw
%   stats(ir).brain.pair(ip).p_fdr
%   stats(ir).brain.pair(ip).p_fwe
%   stats(ir).roi.band_psd
%   stats(ir).roi.pair(ip).p_raw
%   stats(ir).roi.pair(ip).p_fdr
%   stats(ir).roi.pair(ip).p_fwe
%
% Multiple-comparison family:
%   nComp = nRange * nchoosek(nSes,2)
%   - brain level: corrected across all ranges x session-pairs
%   - ROI level  : for each ROI separately, corrected across all ranges x session-pairs

% -------------------- checks --------------------
assert(isstruct(out) && ~isempty(out), 'Input must be a non-empty struct array.');
req = {'power_roi','power_brain'};
for k = 1:numel(req)
    assert(isfield(out, req{k}), 'out must contain field: %s', req{k});
end

nRange = numel(out);

% -------------------- infer sizes from first element --------------------
pb = out(1).power_brain;
pr = out(1).power_roi;

assert(ismatrix(pb), 'power_brain must be [nSubj x nSes].');
[nSubj, nSes] = size(pb);

if ismatrix(pr)
    % allow single-ROI case: power_roi stored as [nSubj x nSes]
    assert(isequal(size(pr), [nSubj, nSes]), ...
        'If power_roi is 2D, it must be [nSubj x nSes] and match power_brain.');
    nROI = 1;
elseif ndims(pr) == 3
    [nROI, nSubj_roi, nSes_roi] = size(pr);
    assert(nSubj_roi == nSubj && nSes_roi == nSes, ...
        'power_roi size must be [nROI x nSubj x nSes], matching power_brain.');
else
    error('power_roi must be either [nROI x nSubj x nSes] or [nSubj x nSes] for single ROI.');
end

% all session pairs
pairs = nchoosek(1:nSes, 2);
nPair = size(pairs, 1);
pair_names = cell(nPair,1);
for ip = 1:nPair
    pair_names{ip} = sprintf('S%d_vs_S%d', pairs(ip,1), pairs(ip,2));
end

% total comparisons determined by labels/ranges and session pairs
nComp = nRange * nPair;

stats = struct([]);

% collect raw p for later correction
P_brain = nan(nRange, nPair);          % brain-level raw p
P_roi   = nan(nROI, nRange, nPair);    % roi-level raw p

for ir = 1:nRange

    % -------------------- per-range safety checks --------------------
    assert(ismatrix(out(ir).power_brain) && size(out(ir).power_brain,1) == nSubj ...
        && size(out(ir).power_brain,2) == nSes, ...
        'out(%d).power_brain must be [nSubj x nSes].', ir);

    band_psd_brain = out(ir).power_brain;   % [nSubj x nSes]
    band_psd_roi   = out(ir).power_roi;

    % allow single ROI stored as 2D
    if ismatrix(band_psd_roi)
        assert(isequal(size(band_psd_roi), [nSubj, nSes]), ...
            'out(%d).power_roi is 2D, so it must be [nSubj x nSes].', ir);
        band_psd_roi = reshape(band_psd_roi, [1, nSubj, nSes]);  % -> [1 x nSubj x nSes]
    elseif ndims(band_psd_roi) == 3
        assert(size(band_psd_roi,1) == nROI && size(band_psd_roi,2) == nSubj ...
            && size(band_psd_roi,3) == nSes, ...
            'out(%d).power_roi must be [nROI x nSubj x nSes].', ir);
    else
        error('out(%d).power_roi must be either [nROI x nSubj x nSes] or [nSubj x nSes].', ir);
    end

    % -------------------- meta --------------------
    if isfield(out,'label') && ~isempty(out(ir).label)
        stats(ir).label = out(ir).label;
    else
        stats(ir).label = sprintf('range_%d', ir);
    end

    if isfield(out,'idx')
        stats(ir).idx = out(ir).idx;
    else
        stats(ir).idx = [];
    end

    % ============================================================
    % 1) Brain-mean results
    % ============================================================
    stats(ir).brain.band_psd = band_psd_brain;

    for ip = 1:nPair
        s1 = pairs(ip,1);
        s2 = pairs(ip,2);

        x = band_psd_brain(:,s1);
        y = band_psd_brain(:,s2);

        [h,p,ci,st] = ttest(x, y);

        stats(ir).brain.pair(ip).name      = pair_names{ip};
        stats(ir).brain.pair(ip).h         = h;
        stats(ir).brain.pair(ip).p_raw     = p;
        stats(ir).brain.pair(ip).ci        = ci;
        stats(ir).brain.pair(ip).t         = st.tstat;
        stats(ir).brain.pair(ip).df        = st.df;
        stats(ir).brain.pair(ip).mean_diff = mean(x - y, 'omitnan');

        P_brain(ir, ip) = p;
    end

    % ============================================================
    % 2) ROI-wise results (vectorized across ROI)
    % ============================================================
    stats(ir).roi.band_psd = band_psd_roi;

    for ip = 1:nPair
        s1 = pairs(ip,1);
        s2 = pairs(ip,2);

        X = band_psd_roi(:,:,s1);  % [nROI x nSubj]
        Y = band_psd_roi(:,:,s2);  % [nROI x nSubj]

        % paired t-test across subjects (dim=2)
        [h_vec, p_vec, ci_mat, st] = ttest(X, Y, 'Dim', 2);

        stats(ir).roi.pair(ip).name      = pair_names{ip};
        stats(ir).roi.pair(ip).h         = h_vec(:);
        stats(ir).roi.pair(ip).p_raw     = p_vec(:);
        stats(ir).roi.pair(ip).ci        = ci_mat;          % [nROI x 2]
        stats(ir).roi.pair(ip).t         = st.tstat(:);
        stats(ir).roi.pair(ip).df        = st.df(:);
        stats(ir).roi.pair(ip).mean_diff = mean(X - Y, 2, 'omitnan');

        P_roi(:, ir, ip) = p_vec(:);
    end
end

% ============================================================
% 3) Multiple-comparison correction
%    nComp = nRange * nPair
% ============================================================

% -------------------- brain-level correction --------------------
p_brain_vec = P_brain(:);

p_brain_fdr_vec = nan(size(p_brain_vec));
valid_brain = ~isnan(p_brain_vec);
if any(valid_brain)
    p_brain_fdr_vec(valid_brain) = mafdr(p_brain_vec(valid_brain), 'BHFDR', true);
end

p_brain_fwe_vec = min(p_brain_vec * nComp, 1);   % Bonferroni/FWE

P_brain_fdr = reshape(p_brain_fdr_vec, [nRange, nPair]);
P_brain_fwe = reshape(p_brain_fwe_vec, [nRange, nPair]);

% -------------------- ROI-level correction --------------------
% For each ROI separately, correct across all ranges x pairs
P_roi_fdr = nan(size(P_roi));
P_roi_fwe = nan(size(P_roi));

for ir = 1:nRange
    for ip = 1:nPair
        stats(ir).brain.pair(ip).p_fdr = P_brain_fdr(ir, ip);
        stats(ir).brain.pair(ip).p_fwe = P_brain_fwe(ir, ip);
    end
end

for iroi = 1:nROI
    p_this = squeeze(P_roi(iroi, :, :));   % [nRange x nPair]
    p_this = p_this(:);                    % nComp x 1

    p_this_fdr = nan(size(p_this));
    valid_roi = ~isnan(p_this);
    if any(valid_roi)
        p_this_fdr(valid_roi) = mafdr(p_this(valid_roi), 'BHFDR', true);
    end

    P_roi_fdr(iroi,:,:) = reshape(p_this_fdr, [1, nRange, nPair]);
    P_roi_fwe(iroi,:,:) = reshape(min(p_this * nComp, 1), [1, nRange, nPair]);
end

for ir = 1:nRange
    for ip = 1:nPair
        stats(ir).roi.pair(ip).p_fdr = squeeze(P_roi_fdr(:, ir, ip));
        stats(ir).roi.pair(ip).p_fwe = squeeze(P_roi_fwe(:, ir, ip));
    end
end

% optional summary info
for ir = 1:nRange
    stats(ir).mcorr.nRange = nRange;
    stats(ir).mcorr.nSes   = nSes;
    stats(ir).mcorr.nPair  = nPair;
    stats(ir).mcorr.nComp  = nComp;
    stats(ir).mcorr.family = 'Correction across ranges x session-pairs; ROI corrected separately per ROI.';
end

end

