function S = jr_balloon_psd_three_states(C, nSubj)
% JR + Balloon: REST / STEADY / UNSTEADY
%
% Output:
%   - PSD before task regression
%   - PSD after regressing out task activation (HRF-convolved stim)
%   - PSD of fitted task activation term
%   - beta maps
%   - raw / residual / activation time series
%
% Notes:
%   - Direct downsample only
%   - No deletion of first point
%   - HRF uses SPM if available

if nargin < 2 || isempty(nSubj)
    nSubj = 1;
end

%% =========================
% Basic parameters
% ==========================
dt    = C.jr.dt;
teq   = C.jr.teq;
tspan = C.jr.tmax;

% Downsample to BOLD TR = 2 s
ds_bold = round(2 / dt);
fs      = 1 / (ds_bold * dt);

% Direct downsample only
ds_time     = @(x) local_downsample_direct(x, ds_bold);
ds_time_vec = @(v) local_downsample_vec_direct(v, ds_bold);

% Stimulus parameters
intensity           = C.jr.stim;
pulse_duration      = 1;
pulse_interval1     = 8;
pulse_min_interval1 = 4;
pulse_max_interval1 = 16;

% PSD parameters
nfft = 512;

% HRF
hrf = get_hrf(dt);

%% =========================
% Subject 1: run serially to get dimensions
% ==========================
baseSeed = 12345;
C0 = C;
C0.jr.verbose = false;
C0.jr.seed    = baseSeed + 1;

% Stim only for display/output
ts_st0 = steady_sti(tspan, 1/dt, pulse_duration, pulse_interval1, intensity);
ts_un0 = unsteady_sti(tspan, 1/dt, pulse_duration, ...
    pulse_min_interval1, pulse_max_interval1, intensity);

stim_steady   = ds_time_vec(ts_st0);
stim_unsteady = ds_time_vec(ts_un0);

% Full stimulus used in model
ts_st_full0 = [zeros(1, round(teq/dt)), ts_st0(1:end-1)];
ts_un_full0 = [zeros(1, round(teq/dt)), ts_un0(1:end-1)];

R1 = run_one_subject(C0, dt, ds_bold, fs, nfft, hrf, ...
    tspan, teq, intensity, pulse_duration, pulse_interval1, ...
    pulse_min_interval1, pulse_max_interval1, ...
    ts_st_full0, ts_un_full0, ds_time);

% Frequency info
f     = R1.f;
nFreq = numel(f);
nROI  = size(R1.psd_rest, 2);
[~, idx01] = min(abs(f - 0.01));
range = idx01:numel(f);

% Time-domain dims
nT_rest     = size(R1.x_rest, 2);
nT_steady   = size(R1.x_steady, 2);
nT_unsteady = size(R1.x_unsteady, 2);

t_rest     = (0:nT_rest-1) / fs;
t_steady   = (0:nT_steady-1) / fs;
t_unsteady = (0:nT_unsteady-1) / fs;

%% =========================
% Preallocate
% ==========================
% PSD
psd_rest_all          = zeros(nFreq, nROI, nSubj);
psd_steady_all        = zeros(nFreq, nROI, nSubj);
psd_unsteady_all      = zeros(nFreq, nROI, nSubj);
psd_steady_post_all   = zeros(nFreq, nROI, nSubj);
psd_unsteady_post_all = zeros(nFreq, nROI, nSubj);
psd_steady_act_all    = zeros(nFreq, nROI, nSubj);
psd_unsteady_act_all  = zeros(nFreq, nROI, nSubj);

% Beta
beta_steady_all   = zeros(nROI, nSubj);
beta_unsteady_all = zeros(nROI, nSubj);

% Time series
x_rest_all         = zeros(nROI, nT_rest, nSubj, 'double');
x_steady_all       = zeros(nROI, nT_steady, nSubj, 'double');
x_unsteady_all     = zeros(nROI, nT_unsteady, nSubj, 'double');
x_steady_res_all   = zeros(nROI, nT_steady, nSubj, 'double');
x_unsteady_res_all = zeros(nROI, nT_unsteady, nSubj, 'double');
x_steady_act_all   = zeros(nROI, nT_steady, nSubj, 'double');
x_unsteady_act_all = zeros(nROI, nT_unsteady, nSubj, 'double');

%% =========================
% Fill subject 1
% ==========================
psd_rest_all(:,:,1)          = R1.psd_rest;
psd_steady_all(:,:,1)        = R1.psd_steady;
psd_unsteady_all(:,:,1)      = R1.psd_unsteady;
psd_steady_post_all(:,:,1)   = R1.psd_steady_post;
psd_unsteady_post_all(:,:,1) = R1.psd_unsteady_post;
psd_steady_act_all(:,:,1)    = R1.psd_steady_act;
psd_unsteady_act_all(:,:,1)  = R1.psd_unsteady_act;

beta_steady_all(:,1)   = R1.beta_steady;
beta_unsteady_all(:,1) = R1.beta_unsteady;

x_rest_all(:,:,1)         = double(R1.x_rest);
x_steady_all(:,:,1)       = double(R1.x_steady);
x_unsteady_all(:,:,1)     = double(R1.x_unsteady);
x_steady_res_all(:,:,1)   = double(R1.x_steady_res);
x_unsteady_res_all(:,:,1) = double(R1.x_unsteady_res);
x_steady_act_all(:,:,1)   = double(R1.x_steady_act);
x_unsteady_act_all(:,:,1) = double(R1.x_unsteady_act);

%% =========================
% Remaining subjects: parfor
% ==========================
if nSubj > 1
    parfor s = 2:nSubj
        C1 = C;
        C1.jr.verbose = false;
        C1.jr.seed    = s;

        Rs = run_one_subject(C1, dt, ds_bold, fs, nfft, hrf, ...
            tspan, teq, intensity, pulse_duration, pulse_interval1, ...
            pulse_min_interval1, pulse_max_interval1, ...
            [], [], ds_time);

        psd_rest_all(:,:,s)          = Rs.psd_rest;
        psd_steady_all(:,:,s)        = Rs.psd_steady;
        psd_unsteady_all(:,:,s)      = Rs.psd_unsteady;
        psd_steady_post_all(:,:,s)   = Rs.psd_steady_post;
        psd_unsteady_post_all(:,:,s) = Rs.psd_unsteady_post;
        psd_steady_act_all(:,:,s)    = Rs.psd_steady_act;
        psd_unsteady_act_all(:,:,s)  = Rs.psd_unsteady_act;

        beta_steady_all(:,s)   = Rs.beta_steady;
        beta_unsteady_all(:,s) = Rs.beta_unsteady;

        x_rest_all(:,:,s)         = double(Rs.x_rest);
        x_steady_all(:,:,s)       = double(Rs.x_steady);
        x_unsteady_all(:,:,s)     = double(Rs.x_unsteady);
        x_steady_res_all(:,:,s)   = double(Rs.x_steady_res);
        x_unsteady_res_all(:,:,s) = double(Rs.x_unsteady_res);
        x_steady_act_all(:,:,s)   = double(Rs.x_steady_act);
        x_unsteady_act_all(:,:,s) = double(Rs.x_unsteady_act);
    end
end

%% =========================
% Probability-normalized PSD
% ==========================
psd_rest_prob          = psd_rest_all(range,:,:)          ./ sum(psd_rest_all(range,:,:), 1);
psd_steady_prob        = psd_steady_all(range,:,:)        ./ sum(psd_steady_all(range,:,:), 1);
psd_unsteady_prob      = psd_unsteady_all(range,:,:)      ./ sum(psd_unsteady_all(range,:,:), 1);
psd_steady_post_prob   = psd_steady_post_all(range,:,:)   ./ sum(psd_steady_post_all(range,:,:), 1);
psd_unsteady_post_prob = psd_unsteady_post_all(range,:,:) ./ sum(psd_unsteady_post_all(range,:,:), 1);
psd_steady_act_prob    = psd_steady_act_all(range,:,:)    ./ sum(psd_steady_act_all(range,:,:), 1);
psd_unsteady_act_prob  = psd_unsteady_act_all(range,:,:)  ./ sum(psd_unsteady_act_all(range,:,:), 1);

%% =========================
% Pack output
% ==========================
S = struct();

S.nSubj = nSubj;
S.fs    = fs;
S.f     = f;
S.range = range;

S.stim = struct( ...
    'steady',   stim_steady, ...
    'unsteady', stim_unsteady);

S.psd = struct();
S.psd.pre = struct( ...
    'rest',     psd_rest_all, ...
    'steady',   psd_steady_all, ...
    'unsteady', psd_unsteady_all);
S.psd.post = struct( ...
    'steady',   psd_steady_post_all, ...
    'unsteady', psd_unsteady_post_all);
S.psd.act = struct( ...
    'steady',   psd_steady_act_all, ...
    'unsteady', psd_unsteady_act_all);

S.psd_prob = struct();
S.psd_prob.pre = struct( ...
    'rest',     psd_rest_prob, ...
    'steady',   psd_steady_prob, ...
    'unsteady', psd_unsteady_prob);
S.psd_prob.post = struct( ...
    'steady',   psd_steady_post_prob, ...
    'unsteady', psd_unsteady_post_prob);
S.psd_prob.act = struct( ...
    'steady',   psd_steady_act_prob, ...
    'unsteady', psd_unsteady_act_prob);

S.beta_task = struct( ...
    'steady',   beta_steady_all, ...
    'unsteady', beta_unsteady_all);

S.t = struct( ...
    'rest',     t_rest, ...
    'steady',   t_steady, ...
    'unsteady', t_unsteady);

S.ts = struct();
S.ts.raw = struct( ...
    'rest',     x_rest_all, ...
    'steady',   x_steady_all, ...
    'unsteady', x_unsteady_all);
S.ts.post = struct( ...
    'steady',   x_steady_res_all, ...
    'unsteady', x_unsteady_res_all);
S.ts.act = struct( ...
    'steady',   x_steady_act_all, ...
    'unsteady', x_unsteady_act_all);

end


%% =========================
% Helper functions
% ==========================

function R = run_one_subject(Ci, dt, ds_bold, fs, nfft, hrf, ...
    tspan, teq, intensity, pulse_duration, pulse_interval1, ...
    pulse_min_interval1, pulse_max_interval1, ...
    ts_st_full_in, ts_un_full_in, ds_time)

% -------- REST --------
[s0_out, ~, ~, ~, ~] = jansenrit_Euler_network(1, [], Ci);
bold_dt = balloon_from_neural(s0_out', dt);
x_rest  = ds_time(bold_dt);

% -------- STEADY stim --------
if isempty(ts_st_full_in)
    ts_st = steady_sti(tspan, 1/dt, pulse_duration, pulse_interval1, intensity);
    ts_st_full = [zeros(1, round(teq/dt)), ts_st(1:end-1)];
else
    ts_st_full = ts_st_full_in;
end

[s0_out, ~, ~, ~, ~] = jansenrit_Euler_network(1, ts_st_full, Ci);
bold_dt  = balloon_from_neural(s0_out', dt);
x_steady = ds_time(bold_dt);

% -------- UNSTEADY stim --------
if isempty(ts_un_full_in)
    ts_un = unsteady_sti(tspan, 1/dt, pulse_duration, ...
        pulse_min_interval1, pulse_max_interval1, intensity);
    ts_un_full = [zeros(1, round(teq/dt)), ts_un(1:end-1)];
else
    ts_un_full = ts_un_full_in;
end

[s0_out, ~, ~, ~, ~] = jansenrit_Euler_network(1, ts_un_full, Ci);
bold_dt    = balloon_from_neural(s0_out', dt);
x_unsteady = ds_time(bold_dt);

% -------- PSD pre --------
[psd_rest, f]   = periodogram(x_rest', [], nfft, fs);
psd_steady      = periodogram(x_steady', [], nfft, fs);
psd_unsteady    = periodogram(x_unsteady', [], nfft, fs);

% -------- Regressors --------
reg_steady_z   = build_hrf_regressor(ts_st_full, size(x_steady,2), ds_bold, hrf);
reg_unsteady_z = build_hrf_regressor(ts_un_full, size(x_unsteady,2), ds_bold, hrf);

% -------- Regression --------
[x_steady_res, x_steady_act, beta_steady] = regress_out_task(x_steady, reg_steady_z);
[x_unsteady_res, x_unsteady_act, beta_unsteady] = regress_out_task(x_unsteady, reg_unsteady_z);

% -------- PSD post / act --------
psd_steady_post   = periodogram(x_steady_res', [], nfft, fs);
psd_unsteady_post = periodogram(x_unsteady_res', [], nfft, fs);
psd_steady_act    = periodogram(x_steady_act', [], nfft, fs);
psd_unsteady_act  = periodogram(x_unsteady_act', [], nfft, fs);

% -------- Pack --------
R = struct();
R.f = f;

R.x_rest         = x_rest;
R.x_steady       = x_steady;
R.x_unsteady     = x_unsteady;
R.x_steady_res   = x_steady_res;
R.x_unsteady_res = x_unsteady_res;
R.x_steady_act   = x_steady_act;
R.x_unsteady_act = x_unsteady_act;

R.psd_rest          = psd_rest;
R.psd_steady        = psd_steady;
R.psd_unsteady      = psd_unsteady;
R.psd_steady_post   = psd_steady_post;
R.psd_unsteady_post = psd_unsteady_post;
R.psd_steady_act    = psd_steady_act;
R.psd_unsteady_act  = psd_unsteady_act;

R.beta_steady   = beta_steady;
R.beta_unsteady = beta_unsteady;

end


function y = local_downsample_direct(x, ds_bold)
% x: nROI x nTime
y = x(:, 1:ds_bold:end);
end


function y = local_downsample_vec_direct(v, ds_bold)
y = v(1:ds_bold:end);
end


function hrf = get_hrf(dt)
% Try SPM HRF first; otherwise use a simple canonical approximation
try
    hrf = spm_hrf(dt);
    hrf = hrf(:);
catch
    t = (0:dt:32)';
    a1 = 6;  b1 = 1;
    a2 = 16; b2 = 1;
    c  = 1/6;

    h1 = gampdf(t, a1, b1);
    h2 = gampdf(t, a2, b2);

    hrf = h1 - c * h2;
    hrf = hrf / max(hrf + eps);
end
end


function reg_z = build_hrf_regressor(ts_full, nBold, dsBold, hrf)
% Match the same rule as BOLD downsampling:
% direct downsample only, no deleting first points

n_dt_target = nBold * dsBold;
ts_full = ts_full(:)';  % row vector

if numel(ts_full) < n_dt_target
    ts_full = [ts_full, zeros(1, n_dt_target - numel(ts_full))];
else
    ts_full = ts_full(1:n_dt_target);
end

reg_dt = conv(ts_full, hrf(:)', 'full');
reg_dt = reg_dt(1:n_dt_target);

reg = reg_dt(1:dsBold:end);
reg = reg(1:nBold);

reg_z = zscore(reg(:));
end


function [Y_res, Y_act, beta_task] = regress_out_task(Y, reg_z)
% Y:     nROI x nT
% reg_z: nT x 1

[nROI, nT] = size(Y);

X = [ones(nT,1), reg_z(:)];

Y_res     = zeros(nROI, nT);
Y_act     = zeros(nROI, nT);
beta_task = zeros(nROI, 1);

for r = 1:nROI
    y = Y(r,:)';
    b = X \ y;

    yhat_task    = X(:,2) * b(2);
    Y_act(r,:)   = yhat_task';
    Y_res(r,:)   = (y - X*b)';
    beta_task(r) = b(2);
end
end