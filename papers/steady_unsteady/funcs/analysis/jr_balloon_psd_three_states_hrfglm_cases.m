function S = jr_balloon_psd_three_states_hrfglm_cases(C, nSubj, opts)
% Output:
%   S.rest.psd / psd_prob
%   S.steady.case1/2/3.{psd,psd_prob,psd_resid,psd_resid_prob}
%   S.unsteady.case1/2/3.{...}
%   All PSD dims: [nFreq x nROI x nSubj]
%
% Requires: spm_hrf, jansenrit_RK2_network, balloon_from_neural,
%           steady_sti, unsteady_sti
%
% opts (optional):
%   opts.useParfor   (true/false) default: true
%   opts.baseSeed    default: 12345
%   opts.nfft        default: 512
%   opts.amp_u       (case2) default: 0.01
%   opts.amp_bold    (case3) default: 1.0
%   opts.intensity   default: 0.1
%   opts.pulse_duration default: 1
%   opts.pulse_interval1 default: 12
%   opts.pulse_min_interval1 default: 4
%   opts.pulse_max_interval1 default: 16
%   opts.rangeMinHz  default: 0.01

if nargin < 2 || isempty(nSubj), nSubj = 1; end
if nargin < 3, opts = struct(); end

% -------- defaults --------
useParfor = getf(opts,'useParfor', true);
baseSeed  = getf(opts,'baseSeed', 12345);
nfft      = getf(opts,'nfft', 512);
amp_u     = getf(opts,'amp_u', 0.01);
amp_bold  = getf(opts,'amp_bold', 1.0);
rangeMinHz= getf(opts,'rangeMinHz', 0.01);

intensity           = getf(opts,'intensity', 0.1);
pulse_duration      = getf(opts,'pulse_duration', 1);
pulse_interval1     = getf(opts,'pulse_interval1', 12);
pulse_min_interval1 = getf(opts,'pulse_min_interval1', 4);
pulse_max_interval1 = getf(opts,'pulse_max_interval1', 16);

% -------- basic params from C --------
dt    = C.jr.dt;
teq   = C.jr.teq;
tspan = C.jr.tspan;

dsBold = round(2/dt);   % TR=2s
TR = dt*dsBold;
fs = 1/TR;

ds_time = @(x) x(:, 1:dsBold:end);   % x: nROI x nTime(dt) -> nROI x nTR
ds_vec  = @(v) v(1:dsBold:end);      % v: 1 x nTime(dt) -> 1 x nTR

% ---- Load SC (kept as you wrote; if JR uses it internally) ----
load('E:\DATA\Steady-unsteady\SC\group_networks_1012.mat')

% ---- HRF at dt ----
hrf = spm_hrf(dt); hrf = hrf(:)';

% ---- common JR options ----
C0 = C;
C0.jr.verbose    = false;
C0.jr.returnBurn = false;
C0.jr.downsamp   = 1;

% ============================================================
% 0) One serial pilot run to get dimensions, f/range, windows
% ============================================================

% build stim dt (with teq pad) once to define reg length
ts_st0 = steady_sti(tspan, 1/dt, pulse_duration, pulse_interval1, intensity);
ts_st0(end) = [];
ts_st_full0 = [zeros(1, round(teq/dt)), ts_st0];

ts_un0 = unsteady_sti(tspan, 1/dt, pulse_duration, pulse_min_interval1, pulse_max_interval1, intensity);
ts_un0(end) = [];
ts_un_full0 = [zeros(1, round(teq/dt)), ts_un0];

% rest pilot
Cpilot = C0; Cpilot.jr.seed = baseSeed + 1;
[y,~,~] = jansenrit_RK2_network(1, [], Cpilot);
u = squeeze(y(:,2,:) - y(:,3,:));        % Ndt x nROI
bold_dt = balloon_from_neural(u', dt);    % nROI x Ndt
x_rest = ds_time(bold_dt);               % nROI x nTR

nTR  = size(x_rest,2);
nROI = size(x_rest,1);

% Welch window based on TR length
win_len  = max(8, round(nTR/8));
win      = hamming(win_len);
noverlap = round(win_len/2);

% frequency axis
[psd_rest_1, f] = pwelch(x_rest', win, noverlap, nfft, fs);  % [nFreq x nROI]
nFreq = length(f);
range = find(f >= rangeMinHz);

% ---- build HRF regressors at TR (steady/unsteady) ----
regTR_steady = build_hrf_reg_tr(ts_st_full0, hrf, dsBold, nTR);
regTR_un     = build_hrf_reg_tr(ts_un_full0, hrf, dsBold, nTR);

X_steady = [ones(nTR,1), zscore(regTR_steady(:))];
X_un     = [ones(nTR,1), zscore(regTR_un(:))];

stim_steady_tr  = ds_vec([ts_st_full0, zeros(1, max(0, nTR*dsBold - numel(ts_st_full0)))]);
stim_unsteady_tr= ds_vec([ts_un_full0, zeros(1, max(0, nTR*dsBold - numel(ts_un_full0)))]);

stim_steady_tr  = stim_steady_tr(1:nTR);
stim_unsteady_tr= stim_unsteady_tr(1:nTR);

% ============================================================
% 1) Pre-allocate all outputs (parfor-friendly)
% ============================================================

% rest
rest_psd      = zeros(nFreq, nROI, nSubj);

% for each state/case store psd and psd_resid
tmpl = @( ) struct( ...
    'psd',            zeros(nFreq, nROI, nSubj), ...
    'psd_prob',       zeros(numel(range), nROI, nSubj), ...
    'psd_resid',      zeros(nFreq, nROI, nSubj), ...
    'psd_resid_prob', zeros(numel(range), nROI, nSubj) );

steady_case1 = tmpl(); steady_case2 = tmpl(); steady_case3 = tmpl();
un_case1     = tmpl(); un_case2     = tmpl(); un_case3     = tmpl();

% ============================================================
% 2) Main loop over subjects (parfor optional)
% ============================================================

loopBody = @(s) run_one_subject( ...
    C0, s, baseSeed, dt, teq, tspan, ...
    pulse_duration, pulse_interval1, pulse_min_interval1, pulse_max_interval1, intensity, ...
    amp_u, amp_bold, ...
    dsBold, ds_time, ...
    win, noverlap, nfft, fs, f, range, ...
    X_steady, X_un, ...
    hrf);

if useParfor && nSubj > 1
    parfor s = 1:nSubj
        R(s) = loopBody(s); %#ok<PFOUS> 
    end
else
    R = repmat(struct(), 1, nSubj);
    for s = 1:nSubj
        R(s) = loopBody(s);
    end
end

% ============================================================
% 3) Collect results
% ============================================================

for s = 1:nSubj
    rest_psd(:,:,s) = R(s).rest_psd;

    steady_case1.psd(:,:,s)       = R(s).steady.case1.psd;
    steady_case1.psd_resid(:,:,s) = R(s).steady.case1.psd_resid;

    steady_case2.psd(:,:,s)       = R(s).steady.case2.psd;
    steady_case2.psd_resid(:,:,s) = R(s).steady.case2.psd_resid;

    steady_case3.psd(:,:,s)       = R(s).steady.case3.psd;
    steady_case3.psd_resid(:,:,s) = R(s).steady.case3.psd_resid;

    un_case1.psd(:,:,s)           = R(s).unsteady.case1.psd;
    un_case1.psd_resid(:,:,s)     = R(s).unsteady.case1.psd_resid;

    un_case2.psd(:,:,s)           = R(s).unsteady.case2.psd;
    un_case2.psd_resid(:,:,s)     = R(s).unsteady.case2.psd_resid;

    un_case3.psd(:,:,s)           = R(s).unsteady.case3.psd;
    un_case3.psd_resid(:,:,s)     = R(s).unsteady.case3.psd_resid;
end

% probability norm
rest_prob = rest_psd(range,:,:) ./ sum(rest_psd(range,:,:),1);

steady_case1.psd_prob       = steady_case1.psd(range,:,:)       ./ sum(steady_case1.psd(range,:,:),1);
steady_case1.psd_resid_prob = steady_case1.psd_resid(range,:,:) ./ sum(steady_case1.psd_resid(range,:,:),1);

steady_case2.psd_prob       = steady_case2.psd(range,:,:)       ./ sum(steady_case2.psd(range,:,:),1);
steady_case2.psd_resid_prob = steady_case2.psd_resid(range,:,:) ./ sum(steady_case2.psd_resid(range,:,:),1);

steady_case3.psd_prob       = steady_case3.psd(range,:,:)       ./ sum(steady_case3.psd(range,:,:),1);
steady_case3.psd_resid_prob = steady_case3.psd_resid(range,:,:) ./ sum(steady_case3.psd_resid(range,:,:),1);

un_case1.psd_prob           = un_case1.psd(range,:,:)           ./ sum(un_case1.psd(range,:,:),1);
un_case1.psd_resid_prob     = un_case1.psd_resid(range,:,:)     ./ sum(un_case1.psd_resid(range,:,:),1);

un_case2.psd_prob           = un_case2.psd(range,:,:)           ./ sum(un_case2.psd(range,:,:),1);
un_case2.psd_resid_prob     = un_case2.psd_resid(range,:,:)     ./ sum(un_case2.psd_resid(range,:,:),1);

un_case3.psd_prob           = un_case3.psd(range,:,:)           ./ sum(un_case3.psd(range,:,:),1);
un_case3.psd_resid_prob     = un_case3.psd_resid(range,:,:)     ./ sum(un_case3.psd_resid(range,:,:),1);

% ============================================================
% 4) Pack outputs
% ============================================================

S = struct();
S.nSubj  = nSubj;
S.fs     = fs;
S.TR     = TR;
S.f      = f;
S.range  = range;

S.stimTR = struct('steady', stim_steady_tr(:), 'unsteady', stim_unsteady_tr(:));
S.regTR  = struct('steady', regTR_steady(:),  'unsteady', regTR_un(:));

S.rest = struct();
S.rest.psd      = rest_psd;
S.rest.psd_prob = rest_prob;

S.steady = struct('case1', steady_case1, 'case2', steady_case2, 'case3', steady_case3);
S.unsteady = struct('case1', un_case1, 'case2', un_case2, 'case3', un_case3);

S.params = struct('amp_u',amp_u,'amp_bold',amp_bold,'nfft',nfft,'baseSeed',baseSeed,'rangeMinHz',rangeMinHz);
end

% ======================================================================
% Per-subject worker
% ======================================================================
function out = run_one_subject(C0, s, baseSeed, dt, teq, tspan, ...
    pulse_duration, pulse_interval1, pulse_min_interval1, pulse_max_interval1, intensity, ...
    amp_u, amp_bold, dsBold, ds_time, ...
    win, noverlap, nfft, fs, f, range, X_steady, X_un, hrf)

C1 = C0;
C1.jr.seed = baseSeed + s;

% -------- build subject-specific stimuli at dt (allows unsteady randomness per subj) --------
ts_st = steady_sti(tspan, 1/dt, pulse_duration, pulse_interval1, intensity);
ts_st(end) = [];
ts_st_full = [zeros(1, round(teq/dt)), ts_st];

ts_un = unsteady_sti(tspan, 1/dt, pulse_duration, pulse_min_interval1, pulse_max_interval1, intensity);
ts_un(end) = [];
ts_un_full = [zeros(1, round(teq/dt)), ts_un];

% -------- REST --------
[y,~,~] = jansenrit_RK2_network(1, [], C1);
u = squeeze(y(:,2,:) - y(:,3,:));
bold_dt = balloon_from_neural(u', dt);
x_rest = ds_time(bold_dt);
psd_rest = pwelch(x_rest', win, noverlap, nfft, fs);

% -------- STEADY: case1 stim in JR --------
[y,~,~] = jansenrit_RK2_network(1, ts_st_full, C1);
u = squeeze(y(:,2,:) - y(:,3,:));
bold_dt = balloon_from_neural(u', dt);
x = ds_time(bold_dt);
[psd, psd_resid] = psd_and_resid(x, X_steady, win, noverlap, nfft, fs);

steady.case1.psd = psd; steady.case1.psd_resid = psd_resid;

% -------- STEADY: case2 add stim to u (JR->Balloon) --------
[y,~,~] = jansenrit_RK2_network(1, [], C1);
u = squeeze(y(:,2,:) - y(:,3,:));                 % Ndt x nROI
u2 = u + amp_u .* ts_st_full(:);                  % broadcast Ndt x 1
bold_dt = balloon_from_neural(u2', dt);
x = ds_time(bold_dt);
[psd, psd_resid] = psd_and_resid(x, X_steady, win, noverlap, nfft, fs);

steady.case2.psd = psd; steady.case2.psd_resid = psd_resid;

% -------- STEADY: case3 add HRF-shape to BOLD (post-balloon) --------
[y,~,~] = jansenrit_RK2_network(1, [], C1);
u = squeeze(y(:,2,:) - y(:,3,:));
bold_dt = balloon_from_neural(u', dt);
x = ds_time(bold_dt);                              % nROI x nTR

% build TR regressor for THIS subject (based on ts_st_full)
nTR = size(x,2);
regTR = build_hrf_reg_tr(ts_st_full, hrf, dsBold, nTR);
x3 = x + amp_bold .* (regTR(:))';                  % add to all ROI
[psd, psd_resid] = psd_and_resid(x3, X_steady, win, noverlap, nfft, fs);

steady.case3.psd = psd; steady.case3.psd_resid = psd_resid;

% -------- UNSTEADY: case1 stim in JR --------
[y,~,~] = jansenrit_RK2_network(1, ts_un_full, C1);
u = squeeze(y(:,2,:) - y(:,3,:));
bold_dt = balloon_from_neural(u', dt);
x = ds_time(bold_dt);
[psd, psd_resid] = psd_and_resid(x, X_un, win, noverlap, nfft, fs);

unsteady.case1.psd = psd; unsteady.case1.psd_resid = psd_resid;

% -------- UNSTEADY: case2 add stim to u (JR->Balloon) --------
[y,~,~] = jansenrit_RK2_network(1, [], C1);
u = squeeze(y(:,2,:) - y(:,3,:));
u2 = u + amp_u .* ts_un_full(:);
bold_dt = balloon_from_neural(u2', dt);
x = ds_time(bold_dt);
[psd, psd_resid] = psd_and_resid(x, X_un, win, noverlap, nfft, fs);

unsteady.case2.psd = psd; unsteady.case2.psd_resid = psd_resid;

% -------- UNSTEADY: case3 add HRF-shape to BOLD (post-balloon) --------
[y,~,~] = jansenrit_RK2_network(1, [], C1);
u = squeeze(y(:,2,:) - y(:,3,:));
bold_dt = balloon_from_neural(u', dt);
x = ds_time(bold_dt);

nTR = size(x,2);
regTR = build_hrf_reg_tr(ts_un_full, hrf, dsBold, nTR);
x3 = x + amp_bold .* (regTR(:))';
[psd, psd_resid] = psd_and_resid(x3, X_un, win, noverlap, nfft, fs);

unsteady.case3.psd = psd; unsteady.case3.psd_resid = psd_resid;

out = struct();
out.rest_psd = psd_rest;
out.steady = steady;
out.unsteady = unsteady;
end

% ======================================================================
% Helpers
% ======================================================================
function [psd, psd_resid] = psd_and_resid(x, X, win, noverlap, nfft, fs)
% x: nROI x nTR
% X: nTR x 2
psd = pwelch(x', win, noverlap, nfft, fs);     % nFreq x nROI

Y = x';                                       % nTR x nROI
beta = X \ Y;                                 % 2 x nROI
Yres = Y - X*(beta);                          % nTR x nROI
x_resid = Yres';                              % nROI x nTR

psd_resid = pwelch(x_resid', win, noverlap, nfft, fs);
end

function regTR = build_hrf_reg_tr(ts_full_dt, hrf_dt, dsBold, nTR)
ts = ts_full_dt(:)';                 % 1 x Ndt
Ndt_target = nTR * dsBold;

if numel(ts) < Ndt_target
    ts = [ts, zeros(1, Ndt_target - numel(ts))];
else
    ts = ts(1:Ndt_target);
end

reg_dt = conv(ts, hrf_dt, 'full');
reg_dt = reg_dt(1:Ndt_target);

regTR = reg_dt(1:dsBold:end);
regTR = regTR(:);
regTR = regTR(1:nTR);
end

function v = getf(S, name, default)
if isfield(S, name)
    v = S.(name);
else
    v = default;
end
end
