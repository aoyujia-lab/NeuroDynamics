function Out = jr_balloon_3injection_hrfreg_psd(C, mode, whichStim, nSubj)
% Multi-subject JR->Balloon simulation + (optional) HRF-regression.
%
% mode:
%   'REST'       : no stimulus, no regression
%   'JR'         : stimulus goes into JR as model input
%   'preBalloon' : stimulus added to neural drive u before Balloon
%   'postBalloon': HRF regressor added to BOLD after Balloon (then regressed out)
%
% whichStim: 'steady' | 'unsteady'   (ignored when mode='REST')
% nSubj    : number of simulated subjects (different seeds; parfor)

%% ---- basic params ----
dt    = C.jr.dt;
teq   = C.jr.teq;
tspan = C.jr.tspan;

dsBold  = round(2/dt);       % TR = 2s
TR      = dt*dsBold;
fs_bold = 1/TR;

% stimulus settings (edit here if needed)
intensity           = 0.1;
pulse_duration      = 1;
pulse_interval1     = 12;
pulse_min_interval1 = 4;
pulse_max_interval1 = 16;

% injection strengths
injScale_u    = 0.1;        % for preBalloon (add to u)
injScale_bold = 0.1;        % for postBalloon (add to BOLD)

% seed base (optional)
baseSeed = 12345;
if isfield(C,'jr') && isfield(C.jr,'baseSeed') && ~isempty(C.jr.baseSeed)
    baseSeed = C.jr.baseSeed;
end

% need HRF only when regression is used
doRegression = ~strcmp(mode,'REST');
if doRegression
    hrf = spm_hrf(dt);
else
    hrf = [];
end

%% ============================================================
% run subject #1 (serial) to get sizes + PSD settings
%% ============================================================
s = 1;
[stim_dt_1, X_1, reg_dt_1, reg_tr_1, x_bold_1, x_resid_1, beta_1, R2_roi_1, R2_mean_1] = ...
    run_one_subject(C, mode, whichStim, s, baseSeed, dt, teq, tspan, dsBold, ...
    intensity, pulse_duration, pulse_interval1, pulse_min_interval1, pulse_max_interval1, ...
    injScale_u, injScale_bold, hrf);

[nROI, nTR] = size(x_bold_1);

% PSD parameters (based on TR-length)
win_len  = max(16, round(nTR/8));
win      = hamming(win_len);
noverlap = round(win_len/2);
nfft     = 512;

[psd_pre_1, f] = pwelch(x_bold_1', win, noverlap, nfft, fs_bold);
if strcmp(mode,'REST')
    psd_post_1 = psd_pre_1;
else
    psd_post_1 = pwelch(x_resid_1', win, noverlap, nfft, fs_bold);
end

range  = find(f >= 0.01);
nFreq  = numel(f);
nRange = numel(range);

psd_pre_prob_1  = psd_pre_1(range,:)  ./ sum(psd_pre_1(range,:),  1);
psd_post_prob_1 = psd_post_1(range,:) ./ sum(psd_post_1(range,:), 1);

%% ---- preallocate (3D = subj) ----
x_bold_all  = zeros(nROI, nTR, nSubj);
x_resid_all = zeros(nROI, nTR, nSubj);
beta_all    = nan(2, nROI, nSubj);

psd_pre_all       = zeros(nFreq,  nROI, nSubj);
psd_post_all      = zeros(nFreq,  nROI, nSubj);
psd_pre_prob_all  = zeros(nRange, nROI, nSubj);
psd_post_prob_all = zeros(nRange, nROI, nSubj);

% ---- NEW: R2 outputs ----
R2_roi_all  = nan(nROI, nSubj);   % per ROI, per subject
R2_mean_all = nan(1, nSubj);      % brain-mean (across ROI), per subject

% fill subj1
x_bold_all(:,:,1)  = x_bold_1;
x_resid_all(:,:,1) = x_resid_1;
beta_all(:,:,1)    = beta_1;

psd_pre_all(:,:,1)       = psd_pre_1;
psd_post_all(:,:,1)      = psd_post_1;
psd_pre_prob_all(:,:,1)  = psd_pre_prob_1;
psd_post_prob_all(:,:,1) = psd_post_prob_1;

R2_roi_all(:,1)  = R2_roi_1;
R2_mean_all(1,1) = R2_mean_1;

%% ============================================================
% remaining subjects (parfor)
%% ============================================================
if nSubj > 1
    parfor s = 2:nSubj
        [~, ~, ~, ~, x_bold, x_resid, beta, R2_roi, R2_mean] = ...
            run_one_subject(C, mode, whichStim, s, baseSeed, dt, teq, tspan, dsBold, ...
            intensity, pulse_duration, pulse_interval1, pulse_min_interval1, pulse_max_interval1, ...
            injScale_u, injScale_bold, hrf);

        psd_pre = pwelch(x_bold', win, noverlap, nfft, fs_bold);
        if strcmp(mode,'REST')
            psd_post = psd_pre;
        else
            psd_post = pwelch(x_resid', win, noverlap, nfft, fs_bold);
        end

        psd_pre_prob  = psd_pre(range,:)  ./ sum(psd_pre(range,:),  1);
        psd_post_prob = psd_post(range,:) ./ sum(psd_post(range,:), 1);

        x_bold_all(:,:,s)  = x_bold;
        x_resid_all(:,:,s) = x_resid;
        beta_all(:,:,s)    = beta;

        psd_pre_all(:,:,s)       = psd_pre;
        psd_post_all(:,:,s)      = psd_post;
        psd_pre_prob_all(:,:,s)  = psd_pre_prob;
        psd_post_prob_all(:,:,s) = psd_post_prob;

        R2_roi_all(:,s)  = R2_roi;
        R2_mean_all(1,s) = R2_mean;
    end
end

%% ---- pack outputs ----
Out = struct();
Out.mode = mode;
Out.whichStim = whichStim;
Out.nSubj = nSubj;
Out.seed_base = baseSeed;

Out.dt = dt;
Out.TR = TR;
Out.fs_bold = fs_bold;

% store subj1 regressor info (REST => empty)
Out.stim_dt = stim_dt_1;
Out.reg_dt  = reg_dt_1;
Out.reg_tr  = reg_tr_1;
Out.X       = X_1;

Out.f = f;
Out.range = range;

Out.x_bold  = x_bold_all;
Out.x_resid = x_resid_all;
Out.beta    = beta_all;

Out.psd_pre       = psd_pre_all;
Out.psd_post      = psd_post_all;
Out.psd_pre_prob  = psd_pre_prob_all;
Out.psd_post_prob = psd_post_prob_all;

% ---- NEW: R2 ----
Out.R2_roi  = R2_roi_all;    % [nROI x nSubj]
Out.R2_mean = R2_mean_all;   % [1 x nSubj]
end

%% ========================================================================
% Helpers
%% ========================================================================

function [stim_dt, X, reg_dt, reg_tr, x_bold, x_resid, beta, R2_roi, R2_mean] = run_one_subject( ...
    C, mode, whichStim, s, baseSeed, dt, teq, tspan, dsBold, ...
    intensity, pulse_duration, pulse_interval1, pulse_min_interval1, pulse_max_interval1, ...
    injScale_u, injScale_bold, hrf)

% subject-specific C
Ck = C;
Ck.jr.verbose = false;
Ck.jr.seed = baseSeed + s;

% ---- stimulus (dt) ----
if strcmp(mode,'REST')
    stim_dt = [];
else
    stimSeed = baseSeed + 1000 + s; % separate stream for stim randomness
    stim_dt = make_stim_dt(whichStim, tspan, teq, dt, ...
        pulse_duration, pulse_interval1, pulse_min_interval1, pulse_max_interval1, intensity, stimSeed);
end

% ---- run JR ----
if strcmp(mode,'JR')
    stim_forJR = stim_dt;
else
    stim_forJR = [];
end

[y, ~, ~] = jansenrit_RK2_network(1, stim_forJR, Ck);
u = squeeze(y(:,2,:) - y(:,3,:)); % Ntotal x nROI

% ---- inject before Balloon (match length to JR output!) ----
if strcmp(mode,'preBalloon')
    stim_u = match_len(stim_dt, size(u,1)); % Ntotal x 1
    u = u + (stim_u * ones(1, size(u,2))) * injScale_u;
end

% ---- Balloon + downsample to TR ----
bold_dt = balloon_from_neural(u', dt);      % nROI x Ntotal
x_bold  = bold_dt(:, 1:dsBold:end);        % nROI x nTR

% ---- regression / postBalloon injection ----
if strcmp(mode,'REST')
    X = [];
    reg_dt = [];
    reg_tr = [];
    x_resid = x_bold;
    beta = nan(2, size(x_bold,1));
    R2_roi = nan(size(x_bold,1), 1);
    R2_mean = nan;
else
    nTR = size(x_bold,2);
    [reg_dt, reg_tr, X] = make_regressors(stim_dt, hrf, dsBold, nTR);

    if strcmp(mode,'postBalloon')
        x_bold = x_bold + (ones(size(x_bold,1),1) * reg_tr(:)') * injScale_bold;
    end

    [x_resid, beta, R2_roi, R2_mean] = regress_out_with_R2(x_bold, X);
end
end

function stim_dt = make_stim_dt(whichStim, tspan, teq, dt, ...
    pulse_duration, pulse_interval1, pulse_min_interval1, pulse_max_interval1, intensity, stimSeed)

pad = zeros(1, round(teq/dt));

if strcmp(whichStim,'unsteady')
    rng(stimSeed, 'twister');
end

switch whichStim
    case 'steady'
        ts = steady_sti(tspan, 1/dt, pulse_duration, pulse_interval1, intensity);
    case 'unsteady'
        ts = unsteady_sti(tspan, 1/dt, pulse_duration, pulse_min_interval1, pulse_max_interval1, intensity);
end

ts(end) = [];
stim_dt = [pad, ts];
end

function [reg_dt, reg_tr, X] = make_regressors(stim_dt, hrf, dsBold, nTR)
n_dt_target = nTR * dsBold;

stim_dt2 = stim_dt(:)';                      % row
if numel(stim_dt2) < n_dt_target
    stim_dt2 = [stim_dt2, zeros(1, n_dt_target-numel(stim_dt2))];
else
    stim_dt2 = stim_dt2(1:n_dt_target);
end

reg_dt = conv(stim_dt2, hrf, 'full');
reg_dt = reg_dt(1:n_dt_target);

reg_tr = downsample(reg_dt, dsBold);
reg_tr = reg_tr(1:nTR);

X = [ones(nTR,1), zscore(reg_tr(:))];
end

function stim_u = match_len(stim_dt, N)
% trim / pad stimulus to length N (column vector)
stim_u = stim_dt(:);
if numel(stim_u) < N
    stim_u = [stim_u; zeros(N-numel(stim_u),1)];
else
    stim_u = stim_u(1:N);
end
end

function [x_resid, beta, R2_roi, R2_mean] = regress_out_with_R2(x_bold, X)
% regress out X from each ROI and compute R2 per ROI
nROI = size(x_bold,1);
nTR  = size(x_bold,2);

x_resid = zeros(size(x_bold));
beta    = zeros(2, nROI);

R2_roi = nan(nROI,1);

for r = 1:nROI
    y = x_bold(r,:)';
    b = X \ y;

    yhat = X*b;
    e    = y - yhat;

    SSE = sum(e.^2);
    SST = sum((y - mean(y)).^2);

    % handle flat signal (SST=0) -> leave NaN
    if SST > 0
        R2_roi(r) = 1 - SSE/SST;
    end

    beta(:,r) = b;
    x_resid(r,:) = e';
end

% brain-mean R2 (simple average across ROI, ignoring NaNs)
R2_mean = mean(R2_roi, 'omitnan');
end
