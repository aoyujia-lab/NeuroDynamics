function [s0_out, x0_out, x1mx2_out, time_vector, info] = ...
    jansenrit_Euler_network(M, stim, C)

% JR network (Euler)
%
% OUTPUT:
%   s0_out      : pyramidal firing rate s0 [time × nnodes]
%   x0_out      : state x0 [time × nnodes]
%   x1mx2_out   : state (x1 - x2) [time × nnodes]

if nargin < 2, stim = []; end
if nargin < 3 || isempty(C), C = struct(); end

% ======================================================================
% Parameters
% ======================================================================

S = C;
if isfield(C,'jr') && isstruct(C.jr)
    S = C.jr;
end

dt          = pick(S,'dt',1e-3);
teq         = pick(S,'teq',60);
tmax        = pick(S,'tmax',600);
downsamp    = pick(S,'downsamp',10);
seed        = pick(S,'seed',0);
verbose     = pick(S,'verbose',true);
returnBurn  = pick(S,'returnBurn',true);

a      = pick(S,'a_vel',100);
ad     = pick(S,'ad_vel',50);
b      = pick(S,'b_vel',50);
p      = pick(S,'p',2);
sigma  = pick(S,'sigma',2);

Ccoupl = pick(S,'C',135);
A      = pick(S,'A',3.25);
B      = pick(S,'B',22);
alpha  = pick(S,'alpha',0.5);
beta   = pick(S,'beta',0);    % paper-style inhibitory gain
G      = pick(S,'G',2.5);     % kept from your original code

e0 = pick(S,'e0',2.5);
v0 = pick(S,'v0',6);
r0 = pick(S,'r0',0.56);
r1 = pick(S,'r1',0.56);
r2 = pick(S,'r2',0.56);

stimTarget = lower(string(pick(S,'stimTarget','p')));

C1 = Ccoupl * pick(S,'C1',1.00);
C2 = Ccoupl * pick(S,'C2',0.80);
C3 = Ccoupl * pick(S,'C3',0.25);
C4 = Ccoupl * pick(S,'C4',0.25);

% ======================================================================
% Sizes
% ======================================================================
M = double(M);
nnodes = size(M,1);

% ======================================================================
% Local normalization of connectivity (paper-style)
% ======================================================================
Mnorm = M;
Mnorm(1:nnodes+1:end) = 0;            % remove self-connections
colsum = sum(Mnorm, 1);               % normalize by in-strength (column sum)
colsum(colsum == 0) = 1;
Mnorm = bsxfun(@rdivide, Mnorm, colsum);

% ======================================================================
% Initial conditions
% ======================================================================
if isfield(S,'ic') && ~isempty(S.ic)
    y_temp = double(S.ic);
else
    ic_vec = [0.131; 0.171; 0.343; 0.21; 3.07; 2.96; 25.36; 2.42];
    y_temp = repmat(ic_vec, 1, nnodes);
end

if seed > 0
    rng(seed,'twister');
else
    rng('shuffle');
end

ttotal = teq + tmax;
Nsim   = floor(ttotal / dt);
Neq    = floor((teq/dt)  / downsamp);
Nmax   = floor((tmax/dt) / downsamp);
Ntotal = Neq + Nmax;

time_vector = (0:(Ntotal-1)).' * (downsamp * dt);

% ======================================================================
% Preallocate outputs
% ======================================================================
s0_out      = zeros(Ntotal, nnodes, 'double');
x0_out      = zeros(Ntotal, nnodes, 'double');
x1mx2_out   = zeros(Ntotal, nnodes, 'double');

% ======================================================================
% Stimulus accessor
% ======================================================================
if isempty(stim)
    getStim = @(t, step) zeros(1,nnodes);
elseif isa(stim,'function_handle')
    getStim = @(t, step) reshape(stim(t, step, nnodes), 1, nnodes);
else
    stim = double(stim);
    if size(stim,1) ~= nnodes
        error('stim must be nnodes x N');
    end
    if size(stim,2) == Nsim
        getStim = @(t, step) stim(:,step).';
    else
        % assume stim sampled at downsamp rate
        getStim = @(t, step) stim(:, min(size(stim,2), 1+floor(step/downsamp))).';
    end
end

stepsPer10s = max(1, round(10/dt));

% initial output
s0_out(1,:)    = double(compute_s0(y_temp));
x0_out(1,:)    = double(y_temp(1,:));
x1mx2_out(1,:) = double(y_temp(2,:) - y_temp(3,:));

% ======================================================================
% MAIN LOOP (Euler)
% ======================================================================
for step = 1:(Nsim-1)
    t = (step-1)*dt;

    u     = getStim(t, step);
    noise = sigma .* randn(1, nnodes);

    % ---- Euler step ----
    dy     = jr_dydt(y_temp, u, noise);
    y_temp = y_temp + dt .* dy;

    % ---- compute outputs ----
    s0  = compute_s0(y_temp);
    x0  = y_temp(1,:);
    x1m = y_temp(2,:) - y_temp(3,:);

    % ---- store (downsamp) ----
    if mod(step, downsamp) == 0
        idx = (step/downsamp) + 1;
        if idx <= Ntotal
            s0_out(idx,:)    = double(s0);
            x0_out(idx,:)    = double(x0);
            x1mx2_out(idx,:) = double(x1m);
        end
    end

    if verbose && mod(step, stepsPer10s) == 0
        fprintf('Elapsed time: %d seconds\n', round(step*dt));
    end
end

% ======================================================================
% Burn-in trimming
% ======================================================================
if ~returnBurn
    keep       = (Neq+1):Ntotal;
    s0_out     = s0_out(keep,:);
    x0_out     = x0_out(keep,:);
    x1mx2_out  = x1mx2_out(keep,:);
    time_vector = time_vector(keep) - time_vector(keep(1));
end

info.nnodes = nnodes;
info.Nsim   = Nsim;
info.Ntotal = Ntotal;

% ======================================================================
% Model equations
% ======================================================================
function dy = jr_dydt(y_state, u_in, noise_in)
    x0  = y_state(1,:);
    x1  = y_state(2,:);
    x2  = y_state(3,:);
    x3  = y_state(4,:);
    v0s = y_state(5,:);
    v1s = y_state(6,:);
    v2s = y_state(7,:);
    v3s = y_state(8,:);

    sigm = @(v, rr) (2*e0) ./ (1 + exp(rr .* (v0 - v)));

    % ---- paper-style long-range input: z_i = sum_j Mnorm_ij * x3_j ----
    z_in = G * (Mnorm * x3.');  z_in = z_in.';

    if stimTarget == "x3"
        z_in = z_in + u_in;
    end

    % ---- paper-style pyramidal input uses z_i ----
    pyr_in   = C2.*x1 - C4.*x2 + Ccoupl.*alpha.*z_in;
    s0_local = sigm(pyr_in, r0);

    x0_dot = v0s;
    v0_dot = A*a .* s0_local - 2*a.*v0s - (a^2).*x0;

    x1_dot = v1s;
    p_eff  = p + noise_in;
    if stimTarget == "p"
        p_eff = p_eff + u_in;
    end

    % ---- paper-style inhibitory gain beta enters here ----
    v1_dot = A*a .* (p_eff + sigm(C1.*x0 - Ccoupl.*beta.*x2, r1)) ...
             - 2*a.*v1s - (a^2).*x1;

    x2_dot = v2s;
    v2_dot = B*b .* sigm(C3.*x0, r2) ...
             - 2*b.*v2s - (b^2).*x2;

    x3_dot = v3s;
    % ---- paper-style x3 driven by same sigmoid output as pyramidal channel ----
    v3_dot = A*ad .* s0_local ...
             - 2*ad.*v3s - (ad^2).*x3;

    dy = [x0_dot; x1_dot; x2_dot; x3_dot; ...
          v0_dot; v1_dot; v2_dot; v3_dot];
end

function s0 = compute_s0(y_state)
    x1 = y_state(2,:);
    x2 = y_state(3,:);
    x3 = y_state(4,:);

    z_in = G * (Mnorm * x3.'); z_in = z_in.';
    pyr_in = C2.*x1 - C4.*x2 + Ccoupl.*alpha.*z_in;

    s0 = (2*e0) ./ (1 + exp(r0 .* (v0 - pyr_in)));
end

end

% ======================================================================
% Utility
% ======================================================================
function v = pick(S,name,default)
if isfield(S,name)
    v = S.(name);
else
    v = default;
end
end