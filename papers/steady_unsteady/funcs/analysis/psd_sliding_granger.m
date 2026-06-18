function stat = psd_sliding_granger_allfreq(psd_out, ranges, varargin)
%PSD_SLIDING_GRANGER_ALLFREQ
%
% Test whether power in ranges{1}+ranges{2} Granger-predicts
% each individual frequency point across sliding windows.
%
% INPUT
%   psd_out : [nFreq x nROI x nWin x nSubj x nSes]
%             or [nFreq x nROI x nWin x nSubj]
%
%   ranges  : cell
%             ranges{1}, ranges{2}: frequency bins used to construct X
%
% OPTIONAL
%   'lag'        : Granger lag order, default = 1
%   'detrend'    : true/false, default = true
%   'zscore'     : true/false, default = true
%   'exclude_X'  : true/false, default = true
%                  whether to exclude ranges{1}+ranges{2} from target frequencies
%   'target_idx' : target frequency indices, default = all non-X frequencies
%   'store_ts'   : true/false, default = false
%
% OUTPUT
%   stat.X_to_Y_GC      : [nFreq x nSubj x nSes]
%   stat.X_to_Y_F       : [nFreq x nSubj x nSes]
%   stat.X_to_Y_p       : [nFreq x nSubj x nSes]
%   stat.X_to_Y_deltaR2 : [nFreq x nSubj x nSes]
%
%   stat.group_GC_mean  : [nFreq x 1]
%   stat.group_GC_t     : [nFreq x 1]
%   stat.group_GC_p     : [nFreq x 1]
%
%   stat.X_ts           : [nWin x nSubj x nSes]
%   stat.Y_ts           : optional, [nWin x nFreq x nSubj x nSes]

%% ---- Optional parameters ----
p = inputParser;
addParameter(p, 'lag', 1);
addParameter(p, 'detrend', true);
addParameter(p, 'zscore', true);
addParameter(p, 'exclude_X', true);
addParameter(p, 'target_idx', []);
addParameter(p, 'store_ts', false);
parse(p, varargin{:});

lag_order = p.Results.lag;
do_detrend = p.Results.detrend;
do_zscore = p.Results.zscore;
exclude_X = p.Results.exclude_X;
target_idx_input = p.Results.target_idx;
store_ts = p.Results.store_ts;

%% ---- Make sure psd_out is 5D ----
if ndims(psd_out) == 4
    psd_out = reshape(psd_out, ...
        size(psd_out, 1), size(psd_out, 2), size(psd_out, 3), size(psd_out, 4), 1);
end

[nFreq, nROI, nWin, nSubj, nSes] = size(psd_out);

%% ---- Predictor frequency indices ----
idx_X = unique([ranges{1}(:); ranges{2}(:)]);

if max(idx_X) > nFreq
    error('ranges{1} or ranges{2} exceeds nFreq.');
end

%% ---- Target frequency indices ----
if isempty(target_idx_input)
    target_idx = 1:nFreq;
else
    target_idx = target_idx_input(:).';
end

if exclude_X
    target_idx = setdiff(target_idx, idx_X);
end

if max(target_idx) > nFreq
    error('target_idx exceeds nFreq.');
end

%% ---- Preallocate ----
X_to_Y_GC      = nan(nFreq, nSubj, nSes);
X_to_Y_F       = nan(nFreq, nSubj, nSes);
X_to_Y_p       = nan(nFreq, nSubj, nSes);
X_to_Y_deltaR2 = nan(nFreq, nSubj, nSes);

X_ts = nan(nWin, nSubj, nSes);

if store_ts
    Y_ts = nan(nWin, nFreq, nSubj, nSes);
else
    Y_ts = [];
end

%% ---- Main loop ----
for isubj = 1:nSubj

    fprintf('Subject %d/%d\n', isubj, nSubj);

    for ises = 1:nSes

        this_psd = psd_out(:, :, :, isubj, ises);  % [nFreq x nROI x nWin]

        if all(isnan(this_psd(:)))
            continue
        end

        %% ---- X: ranges{1}+ranges{2}, whole-brain average ----
        % sum over selected frequencies, then average across ROI
        % output should be [nWin x 1]
        X = squeeze(mean(sum(this_psd(idx_X, :, :), 1, 'omitnan'), 2, 'omitnan'));
        X = X(:);

        X_ts(:, isubj, ises) = X;

        %% ---- Loop over every target frequency ----
        for ifreq = target_idx

            % Y: one frequency point, whole-brain average
            Y = squeeze(mean(this_psd(ifreq, :, :), 2, 'omitnan'));
            Y = Y(:);

            %% ---- Valid windows ----
            valid_idx = ~isnan(X) & ~isnan(Y);

            X_valid = X(valid_idx);
            Y_valid = Y(valid_idx);

            if numel(X_valid) <= 2 * lag_order + 5
                continue
            end

            %% ---- Optional preprocessing ----
            if do_detrend
                X_valid = detrend(X_valid);
                Y_valid = detrend(Y_valid);
            end

            if do_zscore
                if std(X_valid) <= eps || std(Y_valid) <= eps
                    continue
                end

                X_valid = zscore(X_valid);
                Y_valid = zscore(Y_valid);
            end

            if store_ts
                tmp = nan(nWin, 1);
                tmp(valid_idx) = Y_valid;
                Y_ts(:, ifreq, isubj, ises) = tmp;
            end

            %% ---- Granger causality: X -> Y_frequency ----
            [gc_xy, F_xy, p_xy, deltaR2_xy] = local_granger_1d(X_valid, Y_valid, lag_order);

            X_to_Y_GC(ifreq, isubj, ises)      = gc_xy;
            X_to_Y_F(ifreq, isubj, ises)       = F_xy;
            X_to_Y_p(ifreq, isubj, ises)       = p_xy;
            X_to_Y_deltaR2(ifreq, isubj, ises) = deltaR2_xy;

        end
    end
end

%% ---- Group-level statistics for each frequency ----
group_GC_mean = nan(nFreq, 1);
group_GC_t    = nan(nFreq, 1);
group_GC_p    = nan(nFreq, 1);

group_deltaR2_mean = nan(nFreq, 1);
group_deltaR2_t    = nan(nFreq, 1);
group_deltaR2_p    = nan(nFreq, 1);

for ifreq = target_idx

    gc_vec = squeeze(X_to_Y_GC(ifreq, :, :));
    gc_vec = gc_vec(:);
    gc_vec = gc_vec(~isnan(gc_vec));

    if numel(gc_vec) > 2
        group_GC_mean(ifreq) = mean(gc_vec, 'omitnan');
        [~, p_gc, ~, stats_gc] = ttest(gc_vec);
        group_GC_p(ifreq) = p_gc;
        group_GC_t(ifreq) = stats_gc.tstat;
    end

    dR2_vec = squeeze(X_to_Y_deltaR2(ifreq, :, :));
    dR2_vec = dR2_vec(:);
    dR2_vec = dR2_vec(~isnan(dR2_vec));

    if numel(dR2_vec) > 2
        group_deltaR2_mean(ifreq) = mean(dR2_vec, 'omitnan');
        [~, p_dR2, ~, stats_dR2] = ttest(dR2_vec);
        group_deltaR2_p(ifreq) = p_dR2;
        group_deltaR2_t(ifreq) = stats_dR2.tstat;
    end

end

%% ---- Output ----
stat.X_to_Y_GC      = X_to_Y_GC;
stat.X_to_Y_F       = X_to_Y_F;
stat.X_to_Y_p       = X_to_Y_p;
stat.X_to_Y_deltaR2 = X_to_Y_deltaR2;

stat.group_GC_mean = group_GC_mean;
stat.group_GC_t    = group_GC_t;
stat.group_GC_p    = group_GC_p;

stat.group_deltaR2_mean = group_deltaR2_mean;
stat.group_deltaR2_t    = group_deltaR2_t;
stat.group_deltaR2_p    = group_deltaR2_p;

stat.X_ts = X_ts;
stat.Y_ts = Y_ts;

stat.idx_X = idx_X;
stat.target_idx = target_idx;
stat.lag = lag_order;
stat.detrend = do_detrend;
stat.zscore = do_zscore;
stat.exclude_X = exclude_X;

end


function [GC, Fval, pval, deltaR2] = local_granger_1d(X, Y, L)
% Test whether X Granger-causes Y.
%
% Restricted model:
%   Y(t) = past Y
%
% Full model:
%   Y(t) = past Y + past X

X = X(:);
Y = Y(:);

T = numel(Y);

GC = nan;
Fval = nan;
pval = nan;
deltaR2 = nan;

if numel(X) ~= T
    error('X and Y must have the same length.');
end

if T <= 2 * L + 2
    return
end

%% ---- Build lagged matrices ----
Y_current = Y((L+1):T);

Y_lag = nan(T-L, L);
X_lag = nan(T-L, L);

for ilag = 1:L
    Y_lag(:, ilag) = Y((L+1-ilag):(T-ilag));
    X_lag(:, ilag) = X((L+1-ilag):(T-ilag));
end

%% ---- Restricted model: Y past only ----
X_restricted = [ones(T-L, 1), Y_lag];

beta_r = X_restricted \ Y_current;
res_r = Y_current - X_restricted * beta_r;
RSS_r = sum(res_r .^ 2);

%% ---- Full model: Y past + X past ----
X_full = [ones(T-L, 1), Y_lag, X_lag];

beta_f = X_full \ Y_current;
res_f = Y_current - X_full * beta_f;
RSS_f = sum(res_f .^ 2);

%% ---- Granger index ----
if RSS_r > 0 && RSS_f > 0
    GC = log(RSS_r / RSS_f);
end

%% ---- Delta R2 ----
SST = sum((Y_current - mean(Y_current)).^2);

if SST > 0
    R2_r = 1 - RSS_r / SST;
    R2_f = 1 - RSS_f / SST;
    deltaR2 = R2_f - R2_r;
end

%% ---- F-test ----
df1 = L;
df2 = (T - L) - (2 * L + 1);

if df2 > 0 && RSS_f > 0
    Fval = ((RSS_r - RSS_f) / df1) / (RSS_f / df2);
    pval = 1 - fcdf(Fval, df1, df2);
end

end