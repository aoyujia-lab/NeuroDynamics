clear; clc;

%% Setup

addpath('D:\OneDrive\Code\Yujia_lab\papers\steady_unsteady\config');

P = project_paths();
C = params_main();

%% Power redistribution
[psd_out, psd_out_z, psd_out_raw, freq_out] = calculate_psd(P, C);
% psd_out(:,:,:,[1 2 3]) = psd_out(:,:,:,[2 3 1]);

C.psd.slide_win_sec  = 120;  % window length: 240 s
C.psd.slide_step_sec = 4;    % slide step: every 10 s
[psd_out2, psd_out_z2, psd_out_raw2, freq_out2, win_center_sec2, subj_dir2] = calculate_psd_sliding(P, C);
mean_psd_out2_brain = squeeze(mean(psd_out2,2));
mean_psd_out2_brain = squeeze(mean(mean_psd_out2_brain,3));

mean_psd_out2_brain2 = mean_psd_out2_brain(:,:,2);

psd_ses2 = psd_out2(:, :, :, :, 3);
psd_ses2 = reshape(psd_ses2, size(psd_ses2,1), size(psd_ses2,2), size(psd_ses2,3), size(psd_ses2,4), 1);

stat_corr2 = psd_sliding_corr_allfreq(psd_ses2, ranges,'zscore',false,'exclude_X',false,'nPerm',10000,'clusterFormingAlpha', 0.10);
d = diff([false; stat_corr2.sig_cluster(:); false]);

cluster_start = find(d == 1);
cluster_end   = find(d == -1) - 1;

cluster_bounds = [cluster_start, cluster_end];
cluster_freq   = freq_out(cluster_bounds);


% PSD summary across ROIs
mean_psd_brain = squeeze(mean(psd_out, 2));           % [nFreq x nSubj x nSes]
mean_psd_subj  = squeeze(mean(mean_psd_brain, 2));    % [nFreq x nSes]
mean_psd_freq  = squeeze(mean(psd_out, 3));           % [nFreq x nSes]

mean_psd_brain_steady = mean_psd_brain(:,:,3);
steady_power = mean(mean_psd_brain_steady(ranges{2},:),1);

mean_psd_brain_rest = mean_psd_brain(:,:,1);
a = mean_psd_brain_steady - mean_psd_brain_rest;
b = mean(a,2);
steady_power = mean(mean_psd_brain_rest(ranges{2},:),1);

steady_power = mean(mean_psd_brain_steady([ranges{4}],:),1);


B_slope = zeros(1, length(mean_psd_brain_steady));
for i = 1:length(mean_psd_brain_steady)
    [r,p] = corrcoef(steady_power, mean_psd_brain_steady(i,:));
    R(i)       = r(2);
    Pb(i)      = p(2);
    bb         = polyfit(steady_power, mean_psd_brain_steady(i,:), 1);
    B_slope(i) = bb(1);
    [~,p,~,stats] = ttest(mean_psd_brain_steady(i,:), mean_psd_brain_rest(i,:));
    T(i)      = stats.tstat;
    Pval_t(i) = p;
end

[R_obs, P_obs, sig_cluster, stat]       = freq_corr_cluster_perm(steady_power, mean_psd_brain_steady,'clusterFormingAlpha', 0.10);
[T_obs, P_obs, sig_cluster_t, stat_t]   = freq_paired_t_cluster_perm(mean_psd_brain_steady, mean_psd_brain_rest,'clusterFormingAlpha', 0.10);

d = diff([false; sig_cluster(:); false]);

cluster_start = find(d == 1);
cluster_end   = find(d == -1) - 1;

cluster_bounds = [cluster_start, cluster_end];
cluster_freq   = freq_out(cluster_bounds);


T_cor = a_multicorrect(T, Pval_t, C);
R_cor = a_multicorrect(R(1:32), Pb(1:32), C);


steady_power2 = mean(mean_psd_brain_steady(ranges{3},:),1);
[r,p] = corrcoef(steady_power, steady_power2);
scatter(steady_power, steady_power2)

figure;
plot(R, 'LineWidth', 1.5);
hold on;

yline(rcrit,  '--', 'LineWidth', 1.5);
yline(-rcrit, '--', 'LineWidth', 1.5);

% Frequency ranges (index-based)
ranges = {75:77, 118:120, 1:32, 55:246};
range_labels = {
    'steady_0p083'
    'steady_0p125'
    'ultra_slow_0p01_0p05'
    'unsteady_0p625_0p25'
    };


psd_range     = mean_psd_by_range(psd_out,     ranges, range_labels);
psd_range_z   = mean_psd_by_range(psd_out_z,   ranges, range_labels);
psd_range_raw = mean_psd_by_range(psd_out_raw, ranges, range_labels);

stats_raw = psd_range_ttest(psd_range_raw);
stats     = psd_range_ttest(psd_range);

% Multiple correction
nRange = numel(stats);
nPair  = numel(stats(1).roi.pair);
MC     = struct();
path   = 'E:\DATA\Steady-Unsteady\Visualization';
for ir = 1:nRange
    MC(ir).label = stats(ir).label;
    for ip = 1:nPair
        MC(ir).pair(ip) = a_multicorrect( ...
            stats(ir).roi.pair(ip).t, ...
            stats(ir).roi.pair(ip).p_raw, ...
            C);
        roi2cifti_glasser(mean(mean_psd_freq(ranges{ir},:,ip),1)', path, ['power_',range_labels{ir},'_',num2str(ip)]);
        roi2cifti_glasser(MC(ir).pair(ip).X_correct, path, ['ttest_',range_labels{ir},'_',stats(1).roi.pair(ip).name]);
    end
end

all_vals = [];
for i = 1:length(MC)
    for i2 = 1:length(MC(i).pair)
        all_vals = [all_vals; MC(i).pair(i2).X_correct(:)];
    end
end

max_val  = max(all_vals);
min_val  = min(all_vals);
min_pos  = min(all_vals(all_vals > 0));
max_neg  = max(all_vals(all_vals < 0));

%% Machine learning

% ---- SVM classification ----
S_svm = cell(nRange, 1);
cd(P.results.fig)
for ir = 1:nRange
    S_svm{ir}  = psdsvm_anova_range(psd_range_z(ir).power_roi, C);
    % [fig, roc{ir}] = psdsvm_plot_results(S_svm{ir}, P.results.fig, range_labels{ir});
    svm_stat{ir} = psdsvm_permtest(psd_range_z(ir).power_roi, C);
end

close all

delong_tbl = pairwise_delong_multiclass(S_svm, range_labels);

% ---- Behavior prediction ----
RT = calc_rt(P, C);

nBand      = numel(ranges);
alpha_list = [0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 ...
              0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1];

for ia = 19
    ia
    C.ml.enet_alpha_grid = alpha_list(ia);
    for ib = 2
        % S_pred_elas_steady{ia,ib} = predict_behavior_psd_elasticnet( ...
        %     psd_range_z(ib).power_roi(:, :, 2), RT.steady.all.mean(:), [], C);

        P_pred_elas_steady{ia,ib} = permtest_predict_behavior_psd_elasticnet(psd_range_z(ib).power_roi(:, :, 2), RT.steady.all.mean(:), [], C);
        % S_pred_elas_unsteady{ia,ib} = predict_behavior_psd_elasticnet( ...
        %     psd_range_z(ib).power_roi(:, :, 3), RT.unsteady.all.mean(:), [], C);
        % P_pred_elas_unsteady{ia,ib} = permtest_predict_behavior_psd_elasticnet(psd_range_z(ib).power_roi(:, :, 3), RT.unsteady.all.mean(:), [], C);
    end
    % cd(P.results.cache)
    % save('result_qvalue.mat')
end

% Collect Q2 and MSE across alpha/band combinations
for ia = 1:length(alpha_list)
    for ib = 1:4
        Q2_elas_steady(ia,ib)  = S_pred_elas_steady{ia, ib}.eval.Q2_enet;
        % Q2_elas_unsteady(ia,ib) = S_pred_elas_unsteady{ia, ib}.eval.Q2_enet;

        MSE_elas_steady(ia,ib) = S_pred_elas_steady{ia, ib}.eval.mse_enet;
        % MSE_elas_unsteady(ia,ib) = S_pred_elas_unsteady{ia, ib}.eval.mse_enet;
        %
        % P_elas_steady(ia,ib) = S_pred_elas_steady{ia, ib}.eval.p_enet;
        % P_elas_unsteady(ia,ib) = S_pred_elas_unsteady{ia, ib}.eval.p_enet;
        %
        % P_pred_elas_steady{ia, ib}.obs.beta_mean(P_pred_elas_steady{ia, ib}.obs.sel_freq<0.7) = 0;
        % P_pred_elas_steady{ia, ib}.obs.beta_mean(P_pred_elas_steady{ia, ib}.obs.sel_freq<0.7) = 0;
        %
        % beta_elas_steady(ia,ib,:) = P_pred_elas_steady{ia, ib}.obs.beta_mean;
        % beta_elas_unsteady(ia,ib,:) = P_pred_elas_unsteady{ia, ib}.obs.beta_mean;
        %
        % a_multicorrect(P_pred_elas_steady{ia, ib}.obs.beta_mean(label), P_pred_elas_steady{ia, ib}.p.sel_freq_1s(label),C);
        %
        RT_pred_steady(ia,ib,:) = S_pred_elas_steady{ia, ib}.pred.enet;
        % RT_pred_unsteady(ia,ib,:) = S_pred_elas_unsteady{ia, ib}.pred.enet;
    end
end

RT_band1 = squeeze(RT_pred_steady(9,  1, :));
RT_band2 = squeeze(RT_pred_steady(19, 2, :));
RT_band3 = squeeze(RT_pred_steady(10, 3, :));

% roi2cifti_glasser(squeeze(beta_elas_steady(8,1,:)),path,['beta_elas_band1']);
% roi2cifti_glasser(squeeze(beta_elas_steady(10,2,:)),path,['beta_elas_band2']);
% roi2cifti_glasser(squeeze(beta_elas_steady(10,3,:)),path,['beta_elas_band3']);
%
% scatter(squeeze(RT_pred_steady(8,4,:)),RT.steady.all.mean(:))
% std(squeeze(RT_pred_steady(8,4,:)))
% std(RT.steady.all.mean(:))
%
% r = a_multicorrect(squeeze(R_elas_unsteady(:,1,:)),squeeze(P_elas_unsteady(:,1,:)),C);
% r2 = a_multicorrect(R_elas_steady,P_elas_steady,C);


%% Compare with GLM (steady + unsteady)

condList = {'steady', 'unsteady', 'rest_steady', 'rest_unsteady'};
for ic = 1:numel(condList)
    cond = condList{ic};

    % ----------------------------
    % 1) Build design matrix X
    % ----------------------------
    switch cond
        case 'steady'
            steadySpec.L_onsets = 0:12:C.glm.totalDuration;
            steadySpec.R_onsets = 2:8:C.glm.totalDuration;
            X   = build_design_from_onsets_steady(steadySpec.L_onsets, steadySpec.R_onsets, C);
            ses = 'ses-2';

        case 'unsteady'
            path_unsteady  = 'E:\DATA\Steady-unsteady\实验与被试\unsteady_3';  % subfolder: "experiment-and-subjects"
            design_matrix  = build_design_from_xls(path_unsteady);
            X   = design_matrix(11:end,:);
            ses = 'ses-3';

        case 'rest_steady'
            steadySpec.L_onsets = 0:12:C.glm.totalDuration;
            steadySpec.R_onsets = 2:8:C.glm.totalDuration;
            X   = build_design_from_onsets_steady(steadySpec.L_onsets, steadySpec.R_onsets, C);
            ses = 'ses-1';

        case 'rest_unsteady'
            path_unsteady  = 'E:\DATA\Steady-unsteady\实验与被试\unsteady_3';  % subfolder: "experiment-and-subjects"
            design_matrix  = build_design_from_xls(path_unsteady);
            X   = design_matrix(11:end,:);
            ses = 'ses-1';
    end

    % ----------------------------
    % 2) Run GLM
    % ----------------------------
    res          = run_roi_glm_allsubs(P, C, ses, X);
    RES.(cond)   = res;
    X_all.(cond) = X;

    % ----------------------------
    % 3) PSD of GLM residuals
    % ----------------------------
    [PSD_OUT.(cond), FREQ_OUT.(cond)] = compute_psd_zscore(C, res);

    % ----------------------------
    % 4) ROI-wise t-test + correction + visualization
    % ----------------------------
    % Skip activation t-test for rest conditions (beta has no practical meaning)
    if ismember(cond, {'rest_steady', 'rest_unsteady'})
        TSTAT.(cond).beta1 = nan(size(res.raw.beta(:,:,1),1), 1);
        TSTAT.(cond).beta2 = nan(size(res.raw.beta(:,:,2),1), 1);
        PCORR.(cond).beta1 = [];
        PCORR.(cond).beta2 = [];
        continue
    end

    nROI   = size(res.raw.residuals, 2);
    T_beta1 = nan(nROI,1); P_beta1 = nan(nROI,1);
    T_beta2 = nan(nROI,1); P_beta2 = nan(nROI,1);
    for iroi = 1:nROI
        x1 = res.filt.beta(iroi,:, 2);
        [~,p1,~,st1] = ttest(x1);
        T_beta1(iroi) = st1.tstat;
        P_beta1(iroi) = p1;

        x2 = res.filt.beta(iroi,:, 3);
        [~,p2,~,st2] = ttest(x2);
        T_beta2(iroi) = st2.tstat;
        P_beta2(iroi) = p2;
    end

    T_beta1_correct = a_multicorrect(T_beta1, P_beta1, C);
    T_beta2_correct = a_multicorrect(T_beta2, P_beta2, C);

    outPath = 'E:\DATA\Steady-Unsteady\Visualization';
    roi2cifti_glasser(T_beta1_correct.X_correct(:), outPath, sprintf('T08_activation_%s', cond));
    roi2cifti_glasser(T_beta2_correct.X_correct(:), outPath, sprintf('T12_activation_%s', cond));

    TSTAT.(cond).beta1 = T_beta1;
    TSTAT.(cond).beta2 = T_beta2;
    PCORR.(cond).beta1 = T_beta1_correct;
    PCORR.(cond).beta2 = T_beta2_correct;
end


alpha_list = [0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 ...
              0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1];

for ia = 16:numel(alpha_list)
    C.ml.enet_alpha_grid = alpha_list(ia);
    % S_pred_elas_glm_L.(cond){ia} = predict_behavior_psd_elasticnet(RES.steady.raw.beta(:,:,1), RT.steady.all.mean(:), [], C);
    S_pred_elas_glm_R.(cond){ia} = predict_behavior_psd_elasticnet(RES.steady.raw.beta(:,:,2), RT.steady.all.mean(:), [], C);
    % P_pred_elas_glm_L.(cond){ia} = permtest_predict_behavior_psd_elasticnet(res.beta1, y, [], C);
    % P_pred_elas_glm_R.(cond){ia} = permtest_predict_behavior_psd_elasticnet(res.beta2, y, [], C);
end

cd(P.results.cache)
% save('result.mat')


cond_list = {'steady', 'unsteady'};
beta_list = {'beta1', 'beta2'};
ses_idx   = struct('steady', 1, 'unsteady', 2);

all_corr_vals = [];

for irange = 1:4
    for ic = 2
        cond    = cond_list{ic};
        psd_s   = squeeze(mean(psd_out(ranges{irange},:,:,ses_idx.(cond)), 1));

        for ib = 1:2
            r_vec    = nan(360,1); p_vec = nan(360,1);
            beta_mat = RES.(cond).(beta_list{ib});

            for iroi = 1:360
                [r,p]     = corrcoef(beta_mat(iroi,:), psd_s(iroi,:));
                r_vec(iroi) = r(1,2);
                p_vec(iroi) = p(1,2);
            end

            corr_res = a_multicorrect(r_vec, p_vec, C);
            CORR_MAPS.(cond).(beta_list{ib}){irange} = corr_res.X_correct(:);
            all_corr_vals = [all_corr_vals; corr_res.X_correct(:)];

            map_name = sprintf('corr_glm%s_psd%s_%s', beta_list{ib}, ranges{irange}, cond);
            roi2cifti_glasser(corr_res.X_correct(:), outPath, map_name);
        end
    end
end


all_corr_vals = all_corr_vals(~isnan(all_corr_vals));
fprintf('Max:          %.4f\n', max(all_corr_vals));
fprintf('Min:          %.4f\n', min(all_corr_vals));
fprintf('Min positive: %.4f\n', min(all_corr_vals(all_corr_vals > 0)));
fprintf('Max negative: %.4f\n', max(all_corr_vals(all_corr_vals < 0)));


for ia = 1:length(alpha_list)
    Q2_elas_steady_glm2(ia,1) = S_pred_elas_glm_L.rest_unsteady{1, ia}.eval.Q2_enet;
    Q2_elas_steady_glm2(ia,2) = S_pred_elas_glm_R.rest_unsteady{1, ia}.eval.Q2_enet;

    % MSE_elas_unsteady_glm(ia,1) = S_pred_elas_glm_L.unsteady{1, ia}.eval.mse_enet;
    % MSE_elas_unsteady_glm(ia,2) = S_pred_elas_glm_R.unsteady{1, ia}.eval.mse_enet;

    RT_pred_steady_glm2(ia,:,1) = S_pred_elas_glm_L.rest_unsteady{1, ia}.pred.enet;
    RT_pred_steady_glm2(ia,:,2) = S_pred_elas_glm_R.rest_unsteady{1, ia}.pred.enet;
end


x1 = RT_pred_steady_glm(8,:,2)';
x2 = squeeze(RT_pred_steady(19, 2, :));
x3 = squeeze(RT_pred_steady(10, 3, :));

se1 = S_pred_elas_glm_R.rest_unsteady{1, 20}.eval.se_enet;
se2 = S_pred_elas_steady{19, 2}.eval.se_enet;
se3 = S_pred_elas_steady{10, 3}.eval.se_enet;

x11 = RT_pred_steady_glm(17,:,1)';
x22 = squeeze(RT_pred_steady(9, 1, :));

se11 = S_pred_elas_glm_L.rest_unsteady{1, 17}.eval.se_enet;
se22 = S_pred_elas_steady{9, 1}.eval.se_enet;

% Pairwise permutation loss tests
psd_glm_stat1 = paired_perm_loss(x2,  x3,   RT.steady.all.mean(:), 5000);
psd_glm_stat2 = paired_perm_loss(x1,  x11,  RT.steady.all.mean(:), 5000);
psd_glm_stat3 = paired_perm_loss(x1,  x22,  RT.steady.all.mean(:), 5000);
psd_glm_stat4 = paired_perm_loss(x2,  x11,  RT.steady.all.mean(:), 5000);
psd_glm_stat5 = paired_perm_loss(x2,  x22,  RT.steady.all.mean(:), 5000);
psd_glm_stat6 = paired_perm_loss(x11, x22,  RT.steady.all.mean(:), 5000);

P_FDR = mafdr([psd_glm_stat1.p, psd_glm_stat2.p, psd_glm_stat3.p, ...
               psd_glm_stat4.p, psd_glm_stat5.p, psd_glm_stat6.p], 'BHFDR', true);


psd_out_glm = cat(4, PSD_OUT.steady, PSD_OUT.rest_steady, psd_out(:,:,:,1));
psd_range_glm     = mean_psd_by_range(psd_out_glm, ranges, range_labels);
stats_glm         = psd_range_ttest(psd_range_glm);

psd_out_glm_unsteady     = cat(4, PSD_OUT.unsteady, PSD_OUT.rest_unsteady, psd_out(:,:,:,1));
psd_range_glm_unsteady   = mean_psd_by_range(psd_out_glm_unsteady, ranges, range_labels);
stats_glm_unsteady       = psd_range_ttest(psd_range_glm_unsteady);


mean_psd_steady_glm      = mean(mean(PSD_OUT.steady,      2), 3);
mean_psd_rest_steady_glm = mean(mean(PSD_OUT.rest_steady, 2), 3);

mean_psd_unsteady_glm      = mean(mean(PSD_OUT.unsteady,      2), 3);
sd_psd_unsteady_glm        = std(mean_psd_unsteady_glm, 0, 3);
mean_psd_rest_unsteady_glm = mean(mean(PSD_OUT.rest_unsteady, 2), 3);


%% Jansen-Rit + Balloon-Windkessel model

JR = jr_balloon_psd_three_states(C, 50);
psd_rest      = mean(squeeze(JR.psd.pre.rest),    2);
psd_st_pre    = mean(squeeze(JR.psd.pre.steady),  2);
psd_unst_pre  = mean(squeeze(JR.psd_prob.pre.unsteady),  2);
psd_st_post   = mean(squeeze(JR.psd.post.steady), 2);
psd_unst_post = mean(squeeze(JR.psd_prob.post.unsteady), 2);

psd_all      = cat(4, JR.psd_prob.pre.rest, JR.psd_prob.pre.steady,  JR.psd_prob.pre.unsteady);
psd_all_post = cat(4, JR.psd_prob.pre.rest, JR.psd_prob.post.steady, JR.psd_prob.post.unsteady);

% Frequency ranges (index-based)
ranges = {1:32, 118:120, 55:246};
range_labels = {
    'ultra_slow_0p01_0p05'
    'steady_0p125'
    'unsteady_0p625_0p25'
    };

psd_range_jr      = mean_psd_by_range(psd_all,      ranges, range_labels);
psd_range_jr_post = mean_psd_by_range(psd_all_post, ranges, range_labels);

stats      = psd_range_ttest(psd_range_jr);
stats_post = psd_range_ttest(psd_range_jr_post);

% Quick visual check
loglog(psd_rest(11:end))
hold on
loglog(psd_st_pre)
loglog(psd_unst_pre)

loglog(psd_rest)
hold on
loglog(psd_st_post)
loglog(psd_unst_post)

results          = jr_sweep_psd_simple(1, C);
results_unsteady = jr_sweep_psd_simple(1, C, [], [], [], 'unsteady');

range{1} = 1:32;
range{2} = 119;

band = 2;
mean_psd_JR(:,1) = mean(results.a_vel.psd_task(range{band},:),        1);
mean_psd_JR(:,2) = mean(results_unsteady.b_vel.psd_task(range{band},:),1);
mean_psd_JR(:,3) = mean(results_unsteady.A.psd_task(range{band},:),   1);
mean_psd_JR(:,4) = mean(results_unsteady.B.psd_task(range{band},:),   1);
mean_psd_JR(:,5) = mean(results_unsteady.C1.psd_task(range{band},:),  1);
mean_psd_JR(:,6) = mean(results_unsteady.C2.psd_task(range{band},:),  1);
mean_psd_JR(:,7) = mean(results_unsteady.C3.psd_task(range{band},:),  1);
mean_psd_JR(:,8) = mean(results_unsteady.C4.psd_task(range{band},:),  1);

param_JR(:,1) = results.a_vel.param;
param_JR(:,2) = results.b_vel.param;
param_JR(:,3) = results.A.param;
param_JR(:,4) = results.B.param;
param_JR(:,5) = results.C1.param;
param_JR(:,6) = results.C2.param;
param_JR(:,7) = results.C3.param;
param_JR(:,8) = results.C4.param;

for k = 1:8
    [r0104(k), pval0104(k)] = dcor_colwise(mean_psd_JR(:, k), param_JR(:, k));
end

[R2_quad, p_model, p_quad, mdl] = quadratic_r2(x, y);
results.C4.R_task(119)
[R2_quad, p_model, p_quad, mdl] = quadratic_r2(x, y);

scatter(results.a_vel.param, results.a_vel.psd_task(119,:));
a = results.C1.psd_task(119,:)';
b = results.C1.param(:);
plot(results.B.R_task)  % positive correlation

sweepNames = fieldnames(results);
plot_data  = struct();

for si = 1:numel(sweepNames)
    pname = sweepNames{si};

    param_vec = results.(pname).param(:);
    psd_row   = results.(pname).psd_task(119, :).';

    plot_data.(pname) = [param_vec, psd_row];
end


%% Whole-brain network model
load('E:\DATA\Steady-unsteady\SC\averageConnectivity_Fpt.mat')
M = 10.^Fpt;
M(isnan(M)) = 0;
Fpt(isnan(Fpt)) = 0;

intensity       = C.jr.stim;
pulse_duration  = 1;
pulse_interval1 = 8;
dt = C.jr.dt;

ts_st = steady_sti(C.jr.tmax, 1/dt, pulse_duration, pulse_interval1, intensity);
ts_st(end) = [];
ts_st_full = [zeros(1, round(60/dt)), ts_st];

stimM = zeros(360, length(ts_st_full));
rois1 = 1:49;
rois2 = rois1 + 180;
rois  = [rois1, rois2];
other_rois = setdiff(1:360, rois);
stimM(rois, :) = repmat(ts_st_full, length(rois), 1);

%% Parameter sweep settings
nAlpha = 200;
nBeta  = 200;
nR0    = 200;
nSub   = 1;

alphaIndex = linspace(0, 1,   nAlpha);
betaIndex  = linspace(0, 0.5, nBeta);
r0Index    = linspace(0, 1,   nR0);
alpha_fix  = 0.5;
beta_fix   = 0.25;

fs      = 0.5;
ds_bold = round(2 / C.jr.dt);
f_keep  = [];
savefile = 'G:\Yujia_Ao\Data\SSBR\PSD_results_incremental_0321.mat';

PSD_task         = [];
PSD_raw_task     = [];
PSD_task_beta    = [];
PSD_raw_task_beta = [];
PSD_task_r0      = [];
PSD_raw_task_r0  = [];
done_idx_alpha   = 0;
done_idx_beta    = 0;
done_idx_r0      = 0;

sweepConfigs = {
    struct('name', 'alpha', 'values', alphaIndex, ...
           'fixed_alpha', [],        'fixed_beta', beta_fix,  'fixed_r0', C.jr.r0)
    struct('name', 'beta',  'values', betaIndex,  ...
           'fixed_alpha', alpha_fix, 'fixed_beta', [],        'fixed_r0', C.jr.r0)
    struct('name', 'r0',   'values', r0Index,    ...
           'fixed_alpha', alpha_fix, 'fixed_beta', beta_fix,  'fixed_r0', [])
};

for isweep = 1:numel(sweepConfigs)
    cfg    = sweepConfigs{isweep};
    nPoint = numel(cfg.values);

    for i = 1:nPoint
        fprintf('%s  point %d/%d  (%s = %.4f)\n', cfg.name, i, nPoint, cfg.name, cfg.values(i));
        tic

        for sub = 1:nSub
            fprintf('   subject %d/%d\n', sub, nSub);

            Ci          = C;
            Ci.jr.alpha = cfg.fixed_alpha;
            Ci.jr.beta  = cfg.fixed_beta;
            Ci.jr.r0    = cfg.fixed_r0;
            Ci.jr.(cfg.name) = cfg.values(i);
            Ci.jr.seed  = sub;

            [s0_task, ~, ~, ~, ~] = jansenrit_Euler_network(M, stimM, Ci);
            bold_dt_task  = balloon_from_neural(s0_task', C.jr.dt);
            x_steady_task = bold_dt_task(:, 1:ds_bold:end);

            [psd_raw_task_i, f] = periodogram(x_steady_task', [], [], fs);
            idx = (f > 0);
            psd_raw_task_i = psd_raw_task_i(idx, :);

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
            case 'alpha'; done_idx_alpha = i;
            case 'beta';  done_idx_beta  = i;
            case 'r0';    done_idx_r0    = i;
        end

        f_keep = f(idx);
        f_keep(end) = [];

        save(savefile, ...
            'PSD_task',      'PSD_raw_task', ...
            'PSD_task_beta', 'PSD_raw_task_beta', ...
            'PSD_task_r0',   'PSD_raw_task_r0', ...
            'f_keep', 'alphaIndex', 'betaIndex', 'r0Index', ...
            'alpha_fix', 'beta_fix', ...
            'done_idx_alpha', 'done_idx_beta', 'done_idx_r0', ...
            '-v7.3');

        toc
    end
end

psd_region = squeeze(PSD_raw_task_alpha(:, 40, :));
plot(psd_region(128,:))

psd3  = squeeze(mean(PSD_rest_task(:, other_rois, :, :), 2));
psd33 = squeeze(mean(psd3, 3));

psd2  = squeeze(mean(PSD_task_r0(:, other_rois, :, :), 2));
psd22 = squeeze(mean(psd2, 3));

psd1  = squeeze(mean(PSD_task_r0(:, rois, :, :), 2));
psd11 = squeeze(mean(psd1, 3));

scatter(alphaIndex, psd22(128,:))
plot(r0Index,       psd22(128,:))

vis = [1:30,   181:210];
smn = [31:49,  211:229];
co  = [50:76,  230:256];
da  = [77:88,  257:268];
lan = [89:102, 269:282];
fpn = [103:124, 283:304];
aud = [125:132, 305:312];
dmn = [133:172, 313:352];
mmo = [173:177, 353:357];
oa  = [178:180, 358:360];

networks      = {vis, smn, co, da, lan, fpn, aud, dmn, mmo, oa};
network_names = {'vis','smn','co','da','lan','fpn','aud','dmn','mmo','oa'};

freq_idx = 1:32;

%% Normalized PSD per network — alpha sweep
PSD_prob_task_alpha = PSD_raw_task_alpha(11:end, :, :);
PSD_prob_task_alpha = PSD_prob_task_alpha ./ sum(PSD_prob_task_alpha, 1);

psd_alpha = zeros(size(PSD_prob_task_alpha, 3), numel(networks));
for i = 1:numel(networks)
    net         = networks{i};
    psd_alpha(:,i) = squeeze(mean(mean(PSD_prob_task_alpha(freq_idx, net, :), 1), 2));
end

%% Normalized PSD per network — beta sweep
PSD_prob_task_beta = PSD_raw_task_beta(11:end, :, :);
PSD_prob_task_beta = PSD_prob_task_beta ./ sum(PSD_prob_task_beta, 1);

psd_beta = zeros(size(PSD_prob_task_beta, 3), numel(networks));
for i = 1:numel(networks)
    net        = networks{i};
    psd_beta(:,i) = squeeze(mean(mean(PSD_prob_task_beta(freq_idx, net, :), 1), 2));
end

%% Normalized PSD per network — r0 sweep
PSD_prob_task_r0 = PSD_raw_task_r0(11:end, :, :);
PSD_prob_task_r0 = PSD_prob_task_r0 ./ sum(PSD_prob_task_r0, 1);

psd_r0 = zeros(size(PSD_prob_task_r0, 3), numel(networks));
for i = 1:numel(networks)
    net       = networks{i};
    psd_r0(:,i) = squeeze(mean(mean(PSD_prob_task_r0(freq_idx, net, :), 1), 2));
end

name_list = {glasser_L.diminfo{1,2}.maps.table.name};
name_list = name_list(2:end)';

order1 = {'V1','ProS','DVT','MST','V6','V2','V3','V4','V8','V3A','V7','IPS1','FFC','V3B', ...
          'LO1','LO2','PIT','MT','LIPv','VIP','PH','V6A','VMV1','VMV3','V4t','FST','V3CD', ...
          'LO3','VMV2','VVC','4','3b','5m','5L','24dd','24dv','7AL','7PC','1','2','3a', ...
          '6d','6mp','6v','OP4','OP1','OP2-3','FOP2','Ig'}';

% Strip L_ prefix and _ROI suffix, then match to order1
rois_clean = regexprep(name_list, '^L_|_ROI$', '');
[~, idx_in_atlas] = ismember(order1, rois_clean);

% Get corresponding label keys (atlas keys are 0-based)
key_list = {glasser_L.diminfo{1,2}.maps.table.key};
key_list = [key_list{2:end}];

order1_keys        = key_list(idx_in_atlas);
order1_keys_sorted = sort(order1_keys);