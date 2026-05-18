function P = psdsvm_permtest(band_psd, C)
% psdsvm_permtest
% Permutation test for multi-class ECOC SVM classification (AUC & accuracy).
%
% Condition labels are permuted WITHIN each subject to preserve the
% repeated-measures structure, then the full CV + ANOVA-selection pipeline
% (identical to psdsvm_anova_range) is re-run on each permuted dataset.
%
% Uses parfor for speed. Start a pool before calling:
%   parpool('local', N);
%
% INPUT
%   band_psd : [nROI x nSub x nCond]
%   C        : config struct (same as psdsvm_anova_range), reads from C.stats:
%              C.stats.n_perm    — number of permutations (default 1000)
%              C.stats.perm_seed — RNG seed for reproducibility (default 42)
%              C.stats.perm_tail — 'right' (default) | 'two'
%
% OUTPUT (struct P)
%   P.observed      — struct: .AUC [nCond x 1], .acc, .meanAUC
%   P.null_AUC      — [nCond x nPerm]
%   P.null_acc      — [1 x nPerm]
%   P.null_meanAUC  — [1 x nPerm]
%   P.pval_AUC      — [nCond x 1]
%   P.pval_acc      — scalar
%   P.pval_meanAUC  — scalar
%   P.zstat_AUC     — [nCond x 1]
%   P.zstat_acc     — scalar
%   P.CI95_null_AUC — [nCond x 2]
%   P.CI95_null_acc — [1 x 2]
%   P.params        — nPerm, seed, tail
%   P.meta          — nROI, nSub, nCond, nPerm

    % ---------- defaults ----------
    nPerm = getOr(C.stats, 'n_perm',    1000);
    seed  = getOr(C.stats, 'perm_seed', 42);
    tail  = getOr(C.stats, 'perm_tail', 'right');

    [nROI, nSub, nCond] = size(band_psd);

    % ---------- observed ----------
    S_obs   = psdsvm_anova_range(band_psd, C);
    obs_AUC = S_obs.metrics.AUC;
    obs_acc = S_obs.metrics.acc;

    % ---------- pre-generate permutation indices (reproducible) ----------
    % Generate all random permutations on the client with a fixed seed,
    % so results are identical regardless of worker count.
    rng(seed, 'twister');
    perm_idx = zeros(nSub, nCond, nPerm);
    for ip = 1:nPerm
        for s = 1:nSub
            perm_idx(s, :, ip) = randperm(nCond);
        end
    end

    % ---------- pre-build permuted datasets ----------
    % Cell array: parfor can slice cells along the first dimension.
    perm_data = cell(nPerm, 1);
    for ip = 1:nPerm
        tmp = zeros(nROI, nSub, nCond, 'like', band_psd);
        for s = 1:nSub
            tmp(:, s, :) = band_psd(:, s, perm_idx(s, :, ip));
        end
        perm_data{ip} = tmp;
    end

    % ---------- allocate null distributions ----------
    null_AUC = zeros(nCond, nPerm);
    null_acc = zeros(1, nPerm);

    % ---------- progress via DataQueue ----------
    dq    = parallel.pool.DataQueue;
    count = 0;
    t0    = tic;
    afterEach(dq, @(~) progressFcn());

    % ---------- parfor ----------
    C_bc = C;   % broadcast (read-only copy to each worker)

    parfor ip = 1:nPerm
        Sp = psdsvm_anova_range(perm_data{ip}, C_bc);  %#ok<PFBNS>
        null_AUC(:, ip) = Sp.metrics.AUC;
        null_acc(ip)     = Sp.metrics.acc;
        send(dq, ip);
    end

    % ---------- p-values ----------
    null_meanAUC = mean(null_AUC, 1);
    obs_meanAUC  = mean(obs_AUC);

    switch lower(tail)
        case 'right'
            pval_AUC     = (sum(null_AUC >= obs_AUC, 2) + 1) / (nPerm + 1);
            pval_acc     = (sum(null_acc  >= obs_acc)     + 1) / (nPerm + 1);
            pval_meanAUC = (sum(null_meanAUC >= obs_meanAUC) + 1) / (nPerm + 1);
        case 'two'
            mu_auc   = mean(null_AUC, 2);
            pval_AUC = (sum(abs(null_AUC - mu_auc) >= abs(obs_AUC - mu_auc), 2) + 1) / (nPerm + 1);

            mu_acc   = mean(null_acc);
            pval_acc = (sum(abs(null_acc - mu_acc) >= abs(obs_acc - mu_acc)) + 1) / (nPerm + 1);

            mu_mauc      = mean(null_meanAUC);
            pval_meanAUC = (sum(abs(null_meanAUC - mu_mauc) >= abs(obs_meanAUC - mu_mauc)) + 1) / (nPerm + 1);
        otherwise
            error('C.stats.perm_tail must be ''right'' or ''two''.');
    end

    % ---------- z-scores ----------
    zstat_AUC = (obs_AUC - mean(null_AUC, 2)) ./ std(null_AUC, 0, 2);
    zstat_acc = (obs_acc - mean(null_acc))       /  std(null_acc);

    % ---------- 95% null CI ----------
    CI95_null_AUC = prctile(null_AUC', [2.5 97.5])';   % [nCond x 2]
    CI95_null_acc = prctile(null_acc,  [2.5 97.5]);     % [1 x 2]

    % ---------- pack ----------
    P = struct();
    P.observed      = struct('AUC', obs_AUC, 'acc', obs_acc, 'meanAUC', obs_meanAUC);
    P.null_AUC      = null_AUC;
    P.null_acc      = null_acc;
    P.null_meanAUC  = null_meanAUC;
    P.pval_AUC      = pval_AUC;
    P.pval_acc      = pval_acc;
    P.pval_meanAUC  = pval_meanAUC;
    P.zstat_AUC     = zstat_AUC;
    P.zstat_acc     = zstat_acc;
    P.CI95_null_AUC = CI95_null_AUC;
    P.CI95_null_acc = CI95_null_acc;
    P.params        = struct('nPerm', nPerm, 'seed', seed, 'tail', tail);
    P.meta          = struct('nROI', nROI, 'nSub', nSub, 'nCond', nCond, 'nPerm', nPerm);

    fprintf('\nPermutation test done: %d perms, %.1f s\n', nPerm, toc(t0));

    % === nested progress function (runs on client via DataQueue) ===
    function progressFcn()
        count = count + 1;
        if mod(count, 100) == 0 || count == nPerm
            elapsed = toc(t0);
            eta = elapsed / count * (nPerm - count);
            fprintf('  perm %d/%d  (%.1fs elapsed, ETA %.1fs)\n', ...
                count, nPerm, elapsed, eta);
        end
    end
end

% =========================================================================
function v = getOr(s, field, default)
    if isfield(s, field) && ~isempty(s.(field))
        v = s.(field);
    else
        v = default;
    end
end