clear; clc;

%% setup

addpath('D:\OneDrive\Code\Yujia_lab\papers\steady_unsteady\config');

P = project_paths();
C = params_main();


%% Power redistribution

[psd_out, psd_out_z, freq_out] = calculate_psd(P, C);

% psd_out = psd_out .* 100;
psd_out(:,[120,300],:,:)= [];
psd_out_z(:,[120,300],:,:)= [];
% PSD summary across ROIs
mean_psd_brain = squeeze(mean(psd_out, 2));           % [nFreq x nSubj x nSes]
mean_psd_subj  = squeeze(mean(mean_psd_brain, 2));    % [nFreq x nSes]
sd_psd_subj    = squeeze(std(mean_psd_brain, 0, 2));  % [nFreq x nSes]


% Frequency ranges (index-based)
ranges = {1:32, 75:77, 118:120, 55:246};
range_labels = {
    'ultra_slow_0p01_0p05'
    'steady_0p083'
    'steady_0p125'
    'unsteady_0p625_0p25'
    };


psd_range = mean_psd_by_range(psd_out, ranges, range_labels);
psd_range_z = mean_psd_by_range(psd_out_z, ranges, range_labels);
stats     = psd_range_ttest(psd_range);

% multiple correction
nRange = numel(stats);
nPair  = numel(stats(1).roi.pair);
MC = struct();
path = 'E:\DATA\Steady-Unsteady\Visualization';
for ir = 1:nRange
    MC(ir).label = stats(ir).label;
    for ip = 1:nPair
        MC(ir).pair(ip) = a_multicorrect( ...
            stats(ir).roi.pair(ip).t, ...
            stats(ir).roi.pair(ip).p, ...
            C);
        write_roi_scalar_map_cifti(MC(ir).pair(ip).X_correct, path, ['ttest_', range_labels{ir}, '_', stats(1).roi.pair(ip).name]);
    end
end

%% machine learning

% ---- SVM classification ----
svm_tags = {'ultra_slow_', 'steady_0p083', 'steady_0p125', 'task_related'};
S_svm = cell(nRange, 1);
cd(P.results.fig)
for ir = 1:nRange
    S_svm{ir} = psdsvm_anova_range(psd_range_z(ir).power_roi, C);
    [fig, roc{ir}] = psdsvm_plot_results(S_svm{ir}, P.results.fig, svm_tags{ir});
end


close all

delong_tbl = pairwise_delong_multiclass(S_svm, svm_tags);



% ---- Behavior prediction ----
RT = calc_rt(P, C);
nBand = numel(ranges);
% (A) ROI-wise correlation + correction
C_multicorrect.stats.alpha = 0.01;
C_multicorrect.stats.multicorrect = 'NONE';
path = 'E:\DATA\Steady-Unsteady\Visualization';

for ib = 1:nBand
    % steady: session index = 2
    Xs = squeeze(psd_range_z(ib).power_roi(:, :, 2))';   % [nSubj x nROI]
    ys = RT.steady.all.mean(:);

    [r_steady(:, ib), p_steady(:, ib)] = corr(Xs, ys, 'Rows', 'pairwise');
    Rcorr_steady(ib) = a_multicorrect(r_steady(:, ib), p_steady(:, ib), C_multicorrect);
    write_roi_scalar_map_cifti(Rcorr_steady(ib).X_correct, path, ['psd_bhav_connect_steady_band', num2str(nBand)]);

    % unsteady: session index = 3
    Xu = squeeze(psd_range_z(ib).power_roi(:, :, 3))';
    yu = RT.unsteady.all.mean(:);

    [r_unsteady(:, ib), p_unsteady(:, ib)] = corr(Xu, yu, 'Rows', 'pairwise');
    Rcorr_unsteady(ib) = a_multicorrect(r_unsteady(:, ib), p_unsteady(:, ib), C_multicorrect);
    write_roi_scalar_map_cifti(Rcorr_unsteady(ib).X_correct, path, ['psd_bhav_connect_unsteady_band', num2str(nBand)]);
end

% (B) CPM / prediction
alpha_list = [0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];
% alpha_list = [0.9];


for ia = 1:length(alpha_list)
    C.ml.enet_alpha_grid =  alpha_list(ia);

    % S_pred_elas_steady_peak{ia} = predict_behavior_psd_elasticnet( ...
    %     power_peak(:, :, 2), RT.steady.all.mean(:), [], C);
    for ib = 1:4
        % S_pred_steady{ia,ib} = predict_behavior_psd( ...
        %      psd_range_z(ib).power_roi(:, :, 2), RT.steady.all.mean(:), [], C);

        % 
        % S_pred_elas_steady{ia,ib} = predict_behavior_psd_elasticnet( ...
        %     psd_range_z(ib).power_roi(:, :, 2), RT.steady.all.mean(:), [], C);
        % 
        % P_pred_elas_steady{ia,ib} = permtest_predict_behavior_psd_elasticnet(psd_range_z(ib).power_roi(:, :, 2), RT.steady.all.mean(:), [], C, 1000);

        S_pred_elas_unsteady{ia,ib} = predict_behavior_psd_elasticnet( ...
            psd_range_z(ib).power_roi(:, :, 3), RT.unsteady.all.mean(:), [], C);

        P_pred_elas_unsteady{ia,ib} = permtest_predict_behavior_psd_elasticnet(psd_range_z(ib).power_roi(:, :, 3), RT.unsteady.all.mean(:), [], C, 1000);
    end
    save('result.mat')
end

label = 1:360;
label([120,200]) = [];

for ia = 1:length(alpha_list)
    for ib = 1:4
        Q2_elas_steady(ia,ib) = S_pred_elas_steady{ia, ib}.eval.Q2_enet;
        Q2_elas_unsteady(ia,ib) = S_pred_elas_unsteady{ia, ib}.eval.Q2_enet;

        MSE_elas_steady(ia,ib) = S_pred_elas_steady{ia, ib}.eval.mse_enet;
        MSE_elas_unsteady(ia,ib) = S_pred_elas_unsteady{ia, ib}.eval.mse_enet;

        P_elas_steady(ia,ib) = S_pred_elas_steady{ia, ib}.eval.p_enet;
        P_elas_unsteady(ia,ib) = S_pred_elas_unsteady{ia, ib}.eval.p_enet;

        % P_pred_elas_steady{ia, ib}.obs.beta_mean(P_pred_elas_steady{ia, ib}.obs.sel_freq<0.7) = 0;
        % P_pred_elas_steady{ia, ib}.obs.beta_mean(P_pred_elas_steady{ia, ib}.obs.sel_freq<0.7) = 0;



        beta_elas_steady(ia,ib,:) = P_pred_elas_steady{ia, ib}.obs.beta_mean;
        beta_elas_unsteady(ia,ib,:) = P_pred_elas_unsteady{ia, ib}.obs.beta_mean;

        a_multicorrect(P_pred_elas_steady{ia, ib}.obs.beta_mean(label), P_pred_elas_steady{ia, ib}.p.sel_freq_1s(label),C);

        RT_pred_unsteady(ia,ib,:) = S_pred_elas_steady{ia, ib}.pred.enet;
        RT_pred_unsteady(ia,ib,:) = S_pred_elas_unsteady{ia, ib}.pred.enet;
    end
end

RT_band1 = squeeze(beta_elas_steady(10,2,:));
RT_band2 = squeeze(RT_pred_steady(10,2,:));
RT_band3 = squeeze(RT_pred_steady(10,3,:));

write_roi_scalar_map_cifti(squeeze(beta_elas_steady(8,1,:)), path, ['beta_elas_band1']);
write_roi_scalar_map_cifti(squeeze(beta_elas_steady(10,2,:)), path, ['beta_elas_band2']);
write_roi_scalar_map_cifti(squeeze(beta_elas_steady(10,3,:)), path, ['beta_elas_band3']);

scatter(squeeze(RT_pred_steady(8,4,:)),RT.steady.all.mean(:))
std(squeeze(RT_pred_steady(8,4,:)))
std(RT.steady.all.mean(:))

r = a_multicorrect(squeeze(R_elas_unsteady(:,1,:)),squeeze(P_elas_unsteady(:,1,:)),C);
r2 = a_multicorrect(R_elas_steady,P_elas_steady,C);



% Extra model: steady peak features
% S_pred_steady_peak = predict_behavior_psd( ...
%     power_peak(:, :, 2), RT.steady.all.mean(:), [], C);

% scatter(S_pred_elas_steady{1, 3}.pred.enet,RT.steady.all.mean(:))



%% compare with GLM (steady + unsteady)

alpha_list = [0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];

condList = {'steady','unsteady'};

for ic = 1:numel(condList)
    cond = condList{ic};

    % ----------------------------
    % 1) Build design matrix X
    % ----------------------------
    switch cond
        case 'steady'
            % --- Steady pulses: L every 12s from 0; R every 8s from 2
            steadySpec.L_onsets = 0:12:C.glm.totalDuration;
            steadySpec.R_onsets = 2:8:C.glm.totalDuration;

            X = build_design_from_onsets_steady(steadySpec.L_onsets, steadySpec.R_onsets, C);
            ses = 'ses-2';

        case 'unsteady'
            path_unsteady = 'E:\DATA\Steady-unsteady\实验与被试\unsteady_3';
            design_matrix = build_design_from_xls(path_unsteady);

            % 你原来是 [11:end]，保留；如果这里是去掉前10行header/空行就合理
            X = design_matrix(11:end,:);
            ses = 'ses-3';
    end

    % ----------------------------
    % 2) Run GLM
    % ----------------------------
    res = run_roi_glm_allsubs(P, C, ses, X);
    RES.(cond) = res;
    X_all.(cond) = X;

    % ----------------------------
    % 3) PSD (probability)
    % ----------------------------
    [PSD_OUT.(cond), FREQ_OUT.(cond)] = compute_psd_zscore(C, res);

    % ----------------------------
    % 4) ROI-wise ttest + multiple correction + visualization
    % ----------------------------
    nROI = size(res.residuals,2);
    T_beta1 = nan(nROI,1); P_beta1 = nan(nROI,1);
    T_beta2 = nan(nROI,1); P_beta2 = nan(nROI,1);

    for iroi = 1:nROI
        x1 = res.beta1(iroi,:);
        [~,p1,~,st1] = ttest(x1);
        T_beta1(iroi) = st1.tstat;
        P_beta1(iroi) = p1;

        x2 = res.beta2(iroi,:);
        [~,p2,~,st2] = ttest(x2);
        T_beta2(iroi) = st2.tstat;
        P_beta2(iroi) = p2;
    end

    T_beta1_correct = a_multicorrect(T_beta1, P_beta1, C);
    T_beta2_correct = a_multicorrect(T_beta2, P_beta2, C);

    outPath = 'E:\DATA\Steady-Unsteady\Visualization';
    v360_T1 = restore_vector(T_beta1_correct.X_correct(:));
    v360_T2 = restore_vector(T_beta2_correct.X_correct(:));
    write_roi_scalar_map_cifti(v360_T1, outPath, sprintf('T08_activation_%s', cond));
    write_roi_scalar_map_cifti(v360_T2, outPath, sprintf('T12_activation_%s', cond));

    TSTAT.(cond).beta1 = T_beta1;
    TSTAT.(cond).beta2 = T_beta2;
    PCORR.(cond).beta1 = T_beta1_correct;
    PCORR.(cond).beta2 = T_beta2_correct;


    switch cond
        case 'steady'
            y = RT.steady.all.mean(:);
        case 'unsteady'
            % 如果你的 RT 里字段不是这个名字，把它改成你的真实字段
            y = RT.unsteady.all.mean(:);
    end

    % for ia = 1:numel(alpha_list)
    %     C.ml.enet_alpha_grid = alpha_list(ia);
    %     S_pred_elas_glm_L.(cond){ia} = predict_behavior_psd_elasticnet(res.beta1, y, [], C);
    %     S_pred_elas_glm_R.(cond){ia} = predict_behavior_psd_elasticnet(res.beta2, y, [], C);
    %     P_pred_elas_glm_L.(cond){ia} = permtest_predict_behavior_psd_elasticnet(res.beta1, y, [], C, 1000);
    %     P_pred_elas_glm_R.(cond){ia} = permtest_predict_behavior_psd_elasticnet(res.beta2, y, [], C, 1000);
    % end
    % save('result.mat')
end

for ia = 1:length(alpha_list)
    MSE_elas_steady_glm(ia,1) = S_pred_elas_glm_L.steady{1, ia}.eval.mse_enet;
    MSE_elas_steady_glm(ia,2) = S_pred_elas_glm_R.steady{1, ia}.eval.mse_enet;

    MSE_elas_unsteady_glm(ia,1) = S_pred_elas_glm_L.unsteady{1, ia}.eval.mse_enet;
    MSE_elas_unsteady_glm(ia,2) = S_pred_elas_glm_R.unsteady{1, ia}.eval.mse_enet;

    RT_pred_steady_glm(ia,:,1) =  S_pred_elas_glm_L.steady{1, ia}.pred.enet;
    RT_pred_steady_glm(ia,:,2) =  S_pred_elas_glm_R.steady{1, ia}.pred.enet;
end



A = (x2-RT.steady.all.mean(:)).^2;

x1 = RT_pred_steady_glm(10,:,1)';
x2 = squeeze(RT_pred_steady(10,2,:));

psd_glm_stat = paired_perm_loss(x1, x2, RT.steady.all.mean(:), 10000);

scatter(x1,RT.steady.all.mean(:))
  
psd_out_glm = cat(4,psd_out(:,:,:,1),PSD_OUT.steady,PSD_OUT.unsteady);

% Frequency ranges (index-based)
ranges = {1:32, 75:77, 118:120, 55:246};
range_labels = {
    'ultra_slow_0p01_0p05'
    'steady_0p083'
    'steady_0p125'
    'unsteady_0p625_0p25'
    };


psd_range_glm = mean_psd_by_range(psd_out_glm, ranges, range_labels);
stats_glm     = psd_range_ttest(psd_range_glm);


mean_psd_steady_glm = mean(PSD_OUT.steady,2);
mean_psd_steady_glm = mean(mean_psd_steady_glm,3);
sd_psd_steady_glm = std(mean_psd_steady_glm,0,3);


mean_psd_unsteady_glm = mean(PSD_OUT.unsteady,2);
mean_psd_unsteady_glm = mean(mean_psd_unsteady_glm,3);
sd_psd_unsteady_glm = std(mean_psd_unsteady_glm,0,3);

%% compare with FC
[r_mat, p_mat] = calculate_corr_filtered(P, C);
r_mat([120,300],:,:,:) = [];
r_mat(:,[120,300],:,:) = [];


S = edgesvm_anova_cv(r_mat, C);
[fig, roc_peak] = psdsvm_plot_results(S, P.results.fig, 'FC_svm');
% S_pred_fc = predict_behavior_fc(r_mat(:,:,:,2), RT.steady.all.mean(:), [], C);

svm_tags_fc = {'FC', 'steady_0p083', 'steady_0p125'};
S_svm_fc{1} = S;
S_svm_fc{2} = S_svm{2};
S_svm_fc{3} = S_svm{3};


delong_tbl_fc = pairwise_delong_multiclass(S_svm_fc, svm_tags_fc);



for ia = 1:length(alpha_list)
    C.ml.enet_alpha_grid =  alpha_list(ia);
    S_pred_elas_fc{ia} = predict_behavior_psd_elasticnet( ...
        r_mat(:,:,:,2), RT.steady.all.mean(:), [], C);
    
end

for ia = 1:length(alpha_list)
    MSE_elas_steady_fc(ia) = S_pred_elas_fc{ia}.eval.mse_enet;
    RT_pred_steady_fc(ia,:) =  S_pred_elas_fc{ia}.pred.enet;
end


x1 = RT_pred_steady_fc(1,:)';
x2 = squeeze(RT_pred_steady(10,3,:));

A = (x1-RT.steady.all.mean(:)).^2;

psd_fc_stat = paired_perm_loss(x1, x2, RT.steady.all.mean(:), 10000);

% [h,p,ci,stats] = ttest(S_pred_steady_peak.eval.se_neg,S_pred_fc.eval.se_com);
% boxplot([S_pred_steady_peak.eval.se_neg,S_pred_fc.eval.se_com]);

%% JR model

JR = jr_balloon_psd_three_states(C,50);
psd_rest = mean(squeeze(JR.psd_prob.pre.rest),2);
psd_st_pre = mean(squeeze(JR.psd_prob.pre.steady),2);
psd_unst_pre = mean(squeeze(JR.psd_prob.pre.unsteady),2);
psd_st_post = mean(squeeze(JR.psd_prob.post.steady),2);
psd_unst_post = mean(squeeze(JR.psd_prob.post.unsteady),2);

psd_all = cat(4,JR.psd_prob.pre.rest,JR.psd_prob.pre.steady,JR.psd_prob.pre.unsteady);
psd_all_post = cat(4,JR.psd_prob.pre.rest,JR.psd_prob.post.steady,JR.psd_prob.post.unsteady);

% Frequency ranges (index-based)
ranges = {1:32, 118:120, 55:246};
range_labels = {
    'ultra_slow_0p01_0p05'
    'steady_0p125'
    'unsteady_0p625_0p25'
    };


psd_range_jr = mean_psd_by_range(psd_all, ranges, range_labels);

psd_range_jr_post = mean_psd_by_range(psd_all_post, ranges, range_labels);


stats     = psd_range_ttest(psd_range_jr);
stats_post     = psd_range_ttest(psd_range_jr_post);


plot(psd_rest)
hold on
plot(psd_st_pre)
plot(psd_unst_pre)

plot(psd_st_pre)
hold on
plot(psd_st_post)

plot(psd_unst_pre)
hold on
plot(psd_unst_post)

results = jr_sweep_psd_simple(1, C);

scatter(results.B.param,results.B.psd_task(128,:));

plot(results.a_vel.R_task) % positive
plot(results.ad_vel.R_task) % no
plot(results.b_vel.R_task) % strong negative

plot(results.p.R_task) % strong negative
plot(results.sigma.R_task) % strong negative

plot(results.C.R_task) % strong negative
plot(results.C1.R_task) % negative
plot(results.C2.R_task) % negative
plot(results.C3.R_task) % weak positive
plot(results.C4.R_task) %  positive

plot(results.A.R_task) % strong negative
plot(results.B.R_task) %  positive

plot(results.beta.R2) %  positive
plot(results.r0.R2)

plot(results.a_vel.psd_task_minus_rest(118,:))

%% whole brain model
load('E:\DATA\Steady-unsteady\SC\averageConnectivity_Fpt.mat')
M = 10.^Fpt;
M(isnan(M)) = 0;
load('E:\DATA\Steady-unsteady\SC\averageConnectivity_tractLengths.mat')

intensity           = C.jr.stim;
pulse_duration      = 1;
pulse_interval1     = 12;
pulse_min_interval1 = 4;
pulse_max_interval1 = 16;
dt = C.jr.dt;

ts_st = steady_sti(C.jr.tmax, 1/dt, pulse_duration, pulse_interval1, intensity);
ts_st(end) = [];
ts_st_full = [zeros(1, round(60/dt)), ts_st];

stimM = zeros(360,length(ts_st_full));
rois1 = [1:49];
rois2 = rois1+180;
rois = [rois1,rois2];
stimM(rois, :) = repmat(ts_st_full, length(rois), 1);


Gidex = linspace(1,5,10);
p = gcp('nocreate');
if ~isempty(p)
    delete(p);
end

% 开 8 个 worker
parpool('local', 4);

fs = 0.5;
for i = 1:10
    i
    Ci = C;               
    Ci.jr.G = Gidex(i);

    [s0_out, ~, ~, ~, ~] = jansenrit_RK2_network(M, stimM, Ci);

    bold_dt  = balloon_from_neural(s0_out', 2e-3);

    x_steady = downsample(bold_dt', 1000);
    x_steady(1,:) = [];

    [psd, f] = periodogram(x_steady, [], [], fs);

     % ---- 只保留 f > 0.01 Hz ----
    idx = (f > 0.01);
    psd = psd(idx, :);

    % ---- 概率标准化：每个ROI一列，sum over f = 1 ----
    colsum = sum(psd, 1);                 % 1 x nROI
    psd = psd ./ colsum;                  % implicit expansion

    PSD(:,:,i) = psd;
end

psd1 = squeeze(mean(PSD(:,50:end,:),2));
mean_psd = squeeze(mean(PSD,2));
f(1:20) = [];

x_steady = downsample(bold_dt',1000);
x_steady(1,:) = [];

fs = 0.5;
[psd,f] = pwelch(x_steady,[],[],[],fs);
psd2 = psd(7:end,:);

psd08 = psd2(38,:);

mean_psd = mean(psd2,2);

