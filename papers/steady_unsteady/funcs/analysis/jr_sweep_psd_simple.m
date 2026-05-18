function results = jr_sweep_psd_simple(M, C, sweepNames, pct, nSteps, sweepMode)
% 极简 sweep: JR -> Balloon -> downsample -> Periodogram PSD -> corr
%
% sweepMode: 'steady' (default) | 'unsteady' | 'both'
%
% NOTE:
% balloon_from_neural() is assumed to internally remove the first 60 s.
% Therefore, all BOLD/TR lengths here are computed based on the cropped signal.

% ===== defaults =====
if nargin < 3 || isempty(sweepNames)
    sweepNames = {'a_vel','b_vel','C1','C2','C3','C4','A','B'};
    sweepNames = {'a_vel','b_vel'};
end

if nargin < 4 || isempty(pct)
    pct = 0.2;
end

if nargin < 5 || isempty(nSteps)
    nSteps = 500;
end

if nargin < 6 || isempty(sweepMode)
    sweepMode = 'steady';
end

% validate sweepMode
sweepMode = lower(sweepMode);
assert(ismember(sweepMode, {'steady','unsteady','both'}), ...
    'sweepMode must be ''steady'', ''unsteady'', or ''both''.');

do_steady   = ismember(sweepMode, {'steady','both'});
do_unsteady = ismember(sweepMode, {'unsteady','both'});

% ---- stim defaults ----
intensity           = C.jr.stim;
pulse_duration      = 1;
pulse_interval1     = 8;
pulse_min_interval1 = 4;
pulse_max_interval1 = 16;

% ---- sampling/PSD defaults ----
TR   = 2;
fs   = 1 / TR;
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
nBold   = floor(Nsim_bold / ds_bold);

% ---- nodes ----
if isscalar(M)
    nnodes = 1;
else
    nnodes = size(M, 1);
end

% ====== 预生成 stim ======
if do_steady
    ts_steady   = make_steady_pulse(Nsim, dt, pulse_duration, pulse_interval1, intensity);
    stim_steady = [pad, ts_steady];
end

if do_unsteady
    ts_unsteady   = make_unsteady_pulse(Nsim, dt, pulse_duration, ...
                        pulse_min_interval1, pulse_max_interval1, intensity);
    stim_unsteady = [pad, ts_unsteady];
end

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
        pvals = linspace(0, 0.5, nSteps + 1).';
    else
        pvals = linspace(p0 * (1 - pct), p0 * (1 + pct), nSteps + 1).';
    end

    nP = numel(pvals);

    % ---- prealloc (always define both for parfor parser) ----
    x_steady_all   = zeros(nBold, nP);
    x_unsteady_all = zeros(nBold, nP);

    % ---- local copies for parfor: always define both ----
    local_do_steady   = do_steady;
    local_do_unsteady = do_unsteady;
    if do_steady
        local_stim_steady = stim_steady;
    else
        local_stim_steady = [];
    end
    if do_unsteady
        local_stim_unsteady = stim_unsteady;
    else
        local_stim_unsteady = [];
    end

    parfor i1 = 1:nP
        Ci = C;
        Ci.jr.(pname) = pvals(i1);
        Ci.jr.seed    = i1;

        tmp_steady   = nan(nBold, 1);
        tmp_unsteady = nan(nBold, 1);

        % ===== steady =====
        if local_do_steady
            [s0_out, ~, ~, ~, ~] = jansenrit_Euler_network(M, local_stim_steady, Ci);
            u = s0_out;
            if isvector(u), u = u(:); end
            u = u.';
            bold_dt = balloon_from_neural(u, dt);
            idx_tr  = 1:ds_bold:size(bold_dt, 2);
            bold_tr = bold_dt(:, idx_tr).';

            if size(bold_tr, 1) < nBold
                warning('Parameter %s step %d (steady): bold_tr shorter (%d < %d).', ...
                    pname, i1, size(bold_tr,1), nBold);
                tmp_steady(1:size(bold_tr,1)) = bold_tr(:,1);
            else
                tmp_steady = bold_tr(1:nBold, 1);
            end
        end

        % ===== unsteady =====
        if local_do_unsteady
            [s0_out, ~, ~, ~, ~] = jansenrit_Euler_network(M, local_stim_unsteady, Ci);
            u = s0_out;
            if isvector(u), u = u(:); end
            u = u.';
            bold_dt = balloon_from_neural(u, dt);
            idx_tr  = 1:ds_bold:size(bold_dt, 2);
            bold_tr = bold_dt(:, idx_tr).';

            if size(bold_tr, 1) < nBold
                warning('Parameter %s step %d (unsteady): bold_tr shorter (%d < %d).', ...
                    pname, i1, size(bold_tr,1), nBold);
                tmp_unsteady(1:size(bold_tr,1)) = bold_tr(:,1);
            else
                tmp_unsteady = bold_tr(1:nBold, 1);
            end
        end

        if local_do_steady,   x_steady_all(:, i1)   = tmp_steady;   end
        if local_do_unsteady, x_unsteady_all(:, i1) = tmp_unsteady; end
    end

    % ===== post-processing helper =====
    process_psd = @(x_all, pvals_in, label) local_process_psd( ...
        x_all, pvals_in, nfft, fs, pname, label);

    if do_steady
        res_st = process_psd(x_steady_all, pvals, 'steady');
        if isempty(res_st), continue; end
    end

    if do_unsteady
        res_un = process_psd(x_unsteady_all, pvals, 'unsteady');
        if isempty(res_un), continue; end
    end

    % ---- save ----
    if strcmp(sweepMode, 'both')
        % 两种条件都存，用子结构
        results.(pname).steady   = res_st;
        results.(pname).unsteady = res_un;
        results.(pname).param    = pvals;
        results.(pname).nBold    = nBold;
        results.(pname).crop_sec = crop_sec;
    elseif strcmp(sweepMode, 'steady')
        % 保持与原版兼容的平坦结构
        results.(pname).psd_task   = res_st.psd;
        results.(pname).R_task     = res_st.R;
        results.(pname).P_task     = res_st.P;
        results.(pname).P_task_fdr = res_st.P_fdr;
        results.(pname).f          = res_st.f;
        results.(pname).param      = res_st.param;
        results.(pname).nBold      = nBold;
        results.(pname).crop_sec   = crop_sec;
    else  % 'unsteady'
        results.(pname).psd_task   = res_un.psd;
        results.(pname).R_task     = res_un.R;
        results.(pname).P_task     = res_un.P;
        results.(pname).P_task_fdr = res_un.P_fdr;
        results.(pname).f          = res_un.f;
        results.(pname).param      = res_un.param;
        results.(pname).nBold      = nBold;
        results.(pname).crop_sec   = crop_sec;
    end
end

end


% ================= PSD post-processing =================
function res = local_process_psd(x_all, pvals, nfft, fs, pname, label)
% Remove NaN columns, compute periodogram, corr, FDR

valid_col = all(isfinite(x_all), 1);
x_valid   = x_all(:, valid_col);
pvals_v   = pvals(valid_col);

if isempty(pvals_v)
    warning('All simulations invalid for parameter "%s" (%s).', pname, label);
    res = [];
    return;
end

[psd_raw, f] = periodogram(x_valid, [], nfft, fs);

[~, idx01] = min(abs(f - 0.01));
range = idx01:numel(f);

psd_norm = psd_raw(range, :) ./ sum(psd_raw(range, :), 1);
f        = f(range);

[R, P]   = dcor_colwise(psd_norm', pvals_v);
P_fdr    = mafdr(P, 'BHFDR', true);

res = struct();
res.psd   = psd_norm;
res.R     = R;
res.P     = P;
res.P_fdr = P_fdr;
res.f     = f;
res.param = pvals_v;
end


% ================= helpers: stim 生成 =================
function ts = make_steady_pulse(Nsim, dt, dur_s, interval_s, amp)
ts = zeros(1, Nsim);

dur  = max(1, round(dur_s / dt));
step = max(1, round(interval_s / dt));
idx0 = 1:step:Nsim;

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