function S = jr_sweep_psd_task_wholebrain(P,C)
% Sweep JR parameters (alpha, beta, r0) and compute whole-brain BOLD PSD.
%
% This function:
% 1. loads whole-brain structural connectivity
% 2. builds steady stimulation input
% 3. sweeps alpha / beta / r0
% 4. simulates neural activity using jansenrit_Euler_network
% 5. converts neural activity to BOLD using balloon_from_neural
% 6. computes raw PSD and column-normalized PSD
% 7. optionally saves intermediate results after each sweep point
%
% Input
% -----
% C : struct
%     Must contain C.jr and the fields listed below.
%
% Required fields in C.jr
% -----------------------
% dt
% tmax
% stim
% r0
%
% Optional fields in C.jr
% -----------------------
% sc_file           : path to structural connectivity .mat file
% sc_varname        : variable name in sc_file, default 'Fpt'
% savefile          : path to incremental save .mat file, default ''
% nAlpha            : number of alpha sweep points, default 200
% nBeta             : number of beta sweep points,  default 200
% nR0               : number of r0 sweep points,    default 200
% nSub              : number of repetitions,        default 1
% alpha_range       : [min max], default [0 1]
% beta_range        : [min max], default [0 0.5]
% r0_range          : [min max], default [0 1]
% alpha_fix         : fixed alpha when sweeping beta/r0, default 0.5
% beta_fix          : fixed beta when sweeping alpha/r0, default 0.25
% pulse_duration    : stimulation pulse duration (s), default 1
% pulse_interval1   : stimulation pulse interval (s), default 8
% stim_onset        : pre-stim baseline duration (s), default 60
% stim_rois_left    : left hemisphere ROI indices, default 1:49
% stim_rois_right_offset : offset for right hemisphere homologs, default 180
%
% Output
% ------
% S : struct containing PSD results and sweep metadata

%% -------------------- basic settings --------------------
jr = C.jr;

dt        = jr.dt;
tmax      = jr.tmax;
intensity = jr.stim;

sc_file    = local_getfield(jr, 'sc_file', 'G:\Yujia_Ao\Data\SSBR\averageConnectivity_Fpt.mat');
sc_varname = local_getfield(jr, 'sc_varname', 'Fpt');
savefile   = local_getfield(jr, 'savefile', '');

nAlpha = local_getfield(jr, 'nAlpha', 200);
nBeta  = local_getfield(jr, 'nBeta', 200);
nR0    = local_getfield(jr, 'nR0', 200);
nSub   = local_getfield(jr, 'nSub', 1);

alpha_range = local_getfield(jr, 'alpha_range', [0, 1]);
beta_range  = local_getfield(jr, 'beta_range',  [0, 0.5]);
r0_range    = local_getfield(jr, 'r0_range',    [0, 1]);

alpha_fix = local_getfield(jr, 'alpha_fix', 0.5);
beta_fix  = local_getfield(jr, 'beta_fix', 0.25);

pulse_duration  = local_getfield(jr, 'pulse_duration', 1);
pulse_interval1 = local_getfield(jr, 'pulse_interval1', 8);
stim_onset      = local_getfield(jr, 'stim_onset', 60);

stim_rois_left         = local_getfield(jr, 'stim_rois_left', 1:49);
stim_rois_right_offset = local_getfield(jr, 'stim_rois_right_offset', 180);

fs      = 0.5;                  % BOLD sampling frequency after downsampling to TR=2 s
ds_bold = round(2 / dt);

alphaIndex = linspace(alpha_range(1), alpha_range(2), nAlpha);
betaIndex  = linspace(beta_range(1),  beta_range(2),  nBeta);
r0Index    = linspace(r0_range(1),    r0_range(2),    nR0);

%% -------------------- load connectivity --------------------
tmp = load(sc_file);
if ~isfield(tmp, sc_varname)
    error('Variable "%s" not found in file: %s', sc_varname, sc_file);
end

Fpt = tmp.(sc_varname);
M = 10.^Fpt;
M(isnan(M)) = 0;

nROI = size(M, 1);

%% -------------------- build stimulation --------------------
ts_st = steady_sti(tmax, 1 / dt, pulse_duration, pulse_interval1, intensity);
ts_st(end) = [];   % preserve your original behavior

ts_st_full = [zeros(1, round(stim_onset / dt)), ts_st];

stimM = zeros(nROI, length(ts_st_full));
rois_right = stim_rois_left + stim_rois_right_offset;
rois = [stim_rois_left, rois_right];
stimM(rois, :) = repmat(ts_st_full, numel(rois), 1);

%% -------------------- preallocation --------------------
PSD_task          = [];
PSD_raw_task      = [];
PSD_task_beta     = [];
PSD_raw_task_beta = [];
PSD_task_r0       = [];
PSD_raw_task_r0   = [];

done_idx_alpha = 0;
done_idx_beta  = 0;
done_idx_r0    = 0;
f_keep         = [];

sweepConfigs = {
    struct('name', 'alpha', 'values', alphaIndex, ...
           'fixed_alpha', [], 'fixed_beta', beta_fix, 'fixed_r0', jr.r0)
    struct('name', 'beta', 'values', betaIndex, ...
           'fixed_alpha', alpha_fix, 'fixed_beta', [], 'fixed_r0', jr.r0)
    struct('name', 'r0', 'values', r0Index, ...
           'fixed_alpha', alpha_fix, 'fixed_beta', beta_fix, 'fixed_r0', [])
};

%% -------------------- main sweep loop --------------------
for isweep = 1:numel(sweepConfigs)
    cfg = sweepConfigs{isweep};
    nPoint = numel(cfg.values);

    for i = 1:nPoint
        fprintf('%s point %d/%d, %s = %.4f\n', ...
            cfg.name, i, nPoint, cfg.name, cfg.values(i));
        tic

        for sub = 1:nSub
            fprintf('   subject %d/%d\n', sub, nSub);

            Ci = C;
            Ci.jr.alpha = cfg.fixed_alpha;
            Ci.jr.beta  = cfg.fixed_beta;
            Ci.jr.r0    = cfg.fixed_r0;

            Ci.jr.(cfg.name) = cfg.values(i);
            Ci.jr.seed = sub;

            % neural simulation
            [s0_task, ~, ~, ~, ~] = jansenrit_Euler_network(M, stimM, Ci);

            % neural -> BOLD
            bold_dt_task = balloon_from_neural(s0_task', dt);

            % direct point-picking downsampling, same as your original code
            x_steady_task = bold_dt_task(:, 1:ds_bold:end);

            % PSD
            [psd_raw_task_i, f] = periodogram(x_steady_task', [], [], fs);
            idx = (f > 0);

            psd_raw_task_i = psd_raw_task_i(idx, :);

            % column normalization
            colsum_task = sum(psd_raw_task_i, 1);
            colsum_task(colsum_task == 0) = eps;
            psd_norm_task_i = psd_raw_task_i ./ colsum_task;

            switch cfg.name
                case 'alpha'
                    PSD_raw_task(:, :, i, sub) = psd_raw_task_i;
                    PSD_task(:, :, i, sub)     = psd_norm_task_i;

                case 'beta'
                    PSD_raw_task_beta(:, :, i, sub) = psd_raw_task_i;
                    PSD_task_beta(:, :, i, sub)     = psd_norm_task_i;

                case 'r0'
                    PSD_raw_task_r0(:, :, i, sub) = psd_raw_task_i;
                    PSD_task_r0(:, :, i, sub)     = psd_norm_task_i;
            end
        end

        switch cfg.name
            case 'alpha'
                done_idx_alpha = i;
            case 'beta'
                done_idx_beta = i;
            case 'r0'
                done_idx_r0 = i;
        end

        f_keep = f(idx);
        if ~isempty(f_keep)
            f_keep(end) = [];   % preserve your original behavior
        end

        if ~isempty(savefile)
            save(savefile, ...
                'PSD_task', 'PSD_raw_task', ...
                'PSD_task_beta', 'PSD_raw_task_beta', ...
                'PSD_task_r0', 'PSD_raw_task_r0', ...
                'f_keep', 'alphaIndex', 'betaIndex', 'r0Index', ...
                'alpha_fix', 'beta_fix', ...
                'done_idx_alpha', 'done_idx_beta', 'done_idx_r0', ...
                '-v7.3');
        end

        toc
    end
end

%% -------------------- output --------------------
S = struct();

S.PSD_task          = PSD_task;
S.PSD_raw_task      = PSD_raw_task;
S.PSD_task_beta     = PSD_task_beta;
S.PSD_raw_task_beta = PSD_raw_task_beta;
S.PSD_task_r0       = PSD_task_r0;
S.PSD_raw_task_r0   = PSD_raw_task_r0;

S.f_keep = f_keep;

S.alphaIndex = alphaIndex;
S.betaIndex  = betaIndex;
S.r0Index    = r0Index;

S.alpha_fix = alpha_fix;
S.beta_fix  = beta_fix;

S.done_idx_alpha = done_idx_alpha;
S.done_idx_beta  = done_idx_beta;
S.done_idx_r0    = done_idx_r0;

S.M = M;
S.stimM = stimM;

end


function value = local_getfield(s, name, defaultValue)
% Return s.(name) if it exists and is non-empty; otherwise return defaultValue.
if isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = defaultValue;
end
end