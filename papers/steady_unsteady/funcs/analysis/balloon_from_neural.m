function [bold, states] = balloon_from_neural(u, dt, params)
% balloon_from_neural_py
% MATLAB implementation matched to the provided Python algorithm:
% - Euler integration
% - s_dot = rE - (1/taus)*s - (1/tauf)*(f-1)
% - v_dot = (f - v^(1/alpha))*(1/tauo)
% - q_dot = (f*(1-(1-E0)^(1/f))/E0 - q*v^(1/alpha)/v)*(1/tauo)
% - BOLD = V0*(k1*(1-q) + k2*(1-q./v) + k3*(1-v))
%
% Inputs:
%   u:      nROIs x nTime  (neural drive / firing rates)
%   dt:     integration step (seconds)
%   params: struct with fields below
%
% Outputs:
%   bold:   nROIs x nTime_trimmed
%   states: struct with fields s,f,v,q (nROIs x nTime_trimmed)

arguments
    u double
    dt double = 0.01

    % ---- time constants (Python: taus, tauf, tauo) ----
    params.taus double = 0.65   % signal decay (s)
    params.tauf double = 0.41   % feedback regulation (s)
    params.tauo double = 0.98   % venous volume & deoxyHb (s)

    % ---- hemodynamic / MR parameters ----
    params.nu double = 40.3     % frequency offset (s^-1), 1.5T
    params.r0 double = 25       % slope intravascular relaxation vs O2 (s^-1)
    params.alpha double = 0.32  % stiffness
    params.epsilon double = 0.5 % intra/extravascular ratio (Python default 0.5)

    params.E0 double = 0.4      % resting oxygen extraction
    params.TE double = 0.04     % echo time (s)
    params.V0 double = 0.04     % resting venous blood volume fraction

    % ---- numerical safety ----
    params.do_clamp logical = false   % Python code does NOT clamp; keep optional
    params.min_val double = 1e-12

    % ---- burn-in trimming ----
    params.trim_sec double = 60       % remove first 60 s from outputs
end



[nrois, ntime] = size(u);

% ---------- initial conditions (match Python: [0.1, 1, 1, 1]) ----------
s = zeros(nrois, ntime);  s(:,1) = 0.1;
f = ones(nrois, ntime);
v = ones(nrois, ntime);
q = ones(nrois, ntime);

% ---------- constants ----------
itaus = 1 / params.taus;
itauf = 1 / params.tauf;
itauo = 1 / params.tauo;

alpha  = params.alpha;
ialpha = 1 / alpha;

E0 = params.E0;

% ---------- BOLD coefficients (match Python exactly) ----------
k1 = 4.3 * params.nu * E0 * params.TE;
k2 = params.epsilon * params.r0 * E0 * params.TE;
k3 = 1 - params.epsilon;

% ---------- Euler integration ----------
for t = 1:(ntime-1)
    rE = u(:, t);

    st = s(:, t);
    ft = f(:, t);
    vt = v(:, t);
    qt = q(:, t);

    % avoid division / power issues if ft gets too small
    if params.do_clamp
        ft_safe = max(ft, params.min_val);
        vt_safe = max(vt, params.min_val);
    else
        ft_safe = ft;
        vt_safe = vt;
    end

    % s_dot, f_dot
    s_dot = rE - itaus * st - itauf * (ft - 1);
    f_dot = st;

    % v_dot
    v_dot = (ft - vt_safe .^ ialpha) * itauo;

    % q_dot
    Ef_term = 1 - (1 - E0) .^ (1 ./ ft_safe);
    term1   = ft .* Ef_term ./ E0;
    term2   = qt .* (vt_safe .^ (ialpha - 1));
    q_dot   = (term1 - term2) * itauo;

    % update
    s(:, t+1) = st + dt * s_dot;
    f(:, t+1) = ft + dt * f_dot;
    v(:, t+1) = vt + dt * v_dot;
    q(:, t+1) = qt + dt * q_dot;

    if params.do_clamp
        v(:, t+1) = max(v(:, t+1), params.min_val);
        q(:, t+1) = max(q(:, t+1), params.min_val);
        f(:, t+1) = max(f(:, t+1), params.min_val);
    end
end

% ---------- BOLD readout ----------
bold = params.V0 * (k1 * (1 - q) + k2 * (1 - q ./ v) + k3 * (1 - v));

% ---------- trim first trim_sec seconds ----------
nTrim = round(params.trim_sec / dt);

if nTrim >= ntime
    error('trim_sec is too large: all time points would be removed.');
end

keep = (nTrim + 1):ntime;

bold = bold(:, keep);
% bold = zscore(bold,0,2);
states = struct( ...
    's', s(:, keep), ...
    'f', f(:, keep), ...
    'v', v(:, keep), ...
    'q', q(:, keep) ...
);

end