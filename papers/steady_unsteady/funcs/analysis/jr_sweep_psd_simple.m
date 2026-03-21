function results = jr_sweep_psd_simple(M, C, sweepNames, pct, nSteps)
% 极简 sweep: JR -> Balloon -> downsample -> Periodogram PSD -> corr
% NOTE:
% balloon_from_neural() is assumed to internally remove the first 60 s.
% Therefore, all BOLD/TR lengths here are computed based on the cropped signal.

% ===== defaults (内置) =====
if nargin < 3 || isempty(sweepNames)
     sweepNames = {'a_vel','b_vel','C1','C2','C3','C4','A','B'};
    % sweepNames = {'a_vel','b_vel','r0','beta'};
end

if nargin < 4 || isempty(pct)
    pct = 0.2;
end

if nargin < 5 || isempty(nSteps)
    nSteps = 500;
end

% ---- stim defaults ----
intensity       = C.jr.stim;
pulse_duration  = 1;
pulse_interval1 = 8;

% ---- sampling/PSD defaults ----
TR   = 2;
fs   = 1 / TR;   % = 0.5 Hz
nfft = 512;

% ===== base =====
base  = C.jr;
dt    = base.dt;
Nburn = round(base.teq / dt);
Nsim  = round(base.tmax / dt);
pad   = zeros(1, Nburn);

% ===== because balloon_from_neural internally trims first 60 s =====
crop_sec   = 60;
Ncrop      = round(crop_sec / dt);
Nsim_bold  = Nsim - Ncrop;

if Nsim_bold <= 0
    error('After cropping %g s, no samples remain. Check base.tmax and dt.', crop_sec);
end

ds_bold = max(1, round(TR / dt));
nBold   = floor(Nsim_bold / ds_bold);   % TR samples after internal crop

% ---- nodes ----
if isscalar(M)
    nnodes = 1;
else
    nnodes = size(M, 1);
end

% ====== 预生成 stim（只生成一次，省内存省时间）======
ts_task   = make_steady_pulse(Nsim, dt, pulse_duration, pulse_interval1, intensity);
stim_task = [pad, ts_task];    % 1 x (Nburn + Nsim)

results = struct();

for si = 1:numel(sweepNames)
    pname = sweepNames{si}

    if ~isfield(base, pname)
        warning('Skip "%s": not found in C.jr', pname);
        continue;
    end

    % ---- sweep values ----
    p0 = base.(pname);

    if strcmpi(pname, 'beta')
        % beta 特例：固定扫 0 到 0.5
        pvals = linspace(0, 0.5, nSteps + 1).';
    else
        pvals = linspace(p0 * (1 - pct), p0 * (1 + pct), nSteps + 1).';
    end

    nP = numel(pvals);

    % ---- prealloc BOLD(TR) ----
    x_task = zeros(nBold, nP);

    parfor i1 = 1:nP
        Ci = C;
        Ci.jr.(pname) = pvals(i1);
        Ci.jr.seed    = i1;

        % ===== task =====
        [s0_out, ~, ~, ~, ~] = jansenrit_RK2_network(M, stim_task, Ci);

        u = s0_out;   % expected: (T x nnodes) or (T x 1)
        if isvector(u)
            u = u(:);
        end
        u = u.';      % -> (nnodes x T)

        % balloon_from_neural already removes first 60 s internally
        bold_dt = balloon_from_neural(u, dt);   % (nnodes x T_crop)

        % safer than downsample(): explicit indexing
        idx_tr = 1:ds_bold:size(bold_dt, 2);
        bold_tr = bold_dt(:, idx_tr).';         % (Ttr x nnodes)

        if size(bold_tr, 1) < nBold
            warning('Parameter %s step %d: bold_tr shorter than expected (%d < %d). Truncating allocation-dependent length.', ...
                pname, i1, size(bold_tr,1), nBold);
            x_tmp = nan(nBold, 1);
            x_tmp(1:size(bold_tr,1)) = bold_tr(:,1);
            x_task(:, i1) = x_tmp;
        else
            x_task(:, i1) = bold_tr(1:nBold, 1);
        end
    end

    % ===== remove columns with NaN if any =====
    valid_col = all(isfinite(x_task), 1);
    x_task_valid = x_task(:, valid_col);
    pvals_valid  = pvals(valid_col);

    if isempty(pvals_valid)
        warning('All simulations invalid for parameter "%s".', pname);
        continue;
    end

    % ===== Periodogram PSD =====
    [psd_task, f] = periodogram(x_task_valid, [], nfft, fs);

    range = find(f > 0);

    psd_task = psd_task(range, :) ./ sum(psd_task(range, :), 1);
    f        = f(range);

    [R_task, P_task] = dcor_colwise(psd_task', pvals_valid);
    P_task_fdr = mafdr(P_task, 'BHFDR', true);

    % ---- save ----
    results.(pname).psd_task   = psd_task;
    results.(pname).R_task     = R_task;
    results.(pname).P_task     = P_task;
    results.(pname).P_task_fdr = P_task_fdr;
    results.(pname).f          = f;
    results.(pname).param      = pvals_valid;
    results.(pname).nBold      = nBold;
    results.(pname).crop_sec   = crop_sec;
end

end


% ================= helpers: 低内存 stim 生成 =================
function ts = make_steady_pulse(Nsim, dt, dur_s, interval_s, amp)
ts = zeros(1, Nsim);

dur  = max(1, round(dur_s / dt));
step = max(1, round(interval_s / dt));
idx0 = 1:step:Nsim;   % pulse onset indices

for k = 1:numel(idx0)
    a = idx0(k);
    b = min(Nsim, a + dur - 1);
    ts(a:b) = amp;
end

end


function ts = make_unsteady_pulse(Nsim, dt, dur_s, minInt_s, maxInt_s, amp)
ts = zeros(1, Nsim);

dur     = max(1, round(dur_s / dt));
minStep = max(1, round(minInt_s / dt));
maxStep = max(minStep, round(maxInt_s / dt));

t = 1;
while t <= Nsim
    b = min(Nsim, t + dur - 1);
    ts(t:b) = amp;
    t = t + randi([minStep, maxStep], 1, 1);
end

end