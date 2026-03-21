function C = params_main()
%PARAMS_MAIN  Main parameter set for steady_unsteady paper
%
% Usage:
%   C = params_main();

%% ===== 基本信息（不影响结果，但非常有用） =====
C.project.name    = 'steady_unsteady';
C.project.version = 'main_v1';
C.project.date    = datestr(now);

%% ===== 数据相关参数 =====
C.data.TR        = 2.0;          % seconds
C.data.FS        = 1/C.data.TR;          % Hz
C.data.nTR       = 290;          % 
C.data.nROI        = 360;          % 
C.data.excludesubjects  = [2,35];        % or a vector: [1 3 5]
C.data.excluderoi  = [120,200];
C.data.sesnum = 3;


%% ===== 频谱分析参数 =====
C.psd.method    = 'periodogram';
C.psd.win_sec   = ones(240,1);           % window length in seconds
C.psd.overlap   = 120;           % fraction
C.psd.nfft   = 512;

%% ===== 机器学习参数 =====
C.ml.alpha    = 0.01;
C.ml.Kfallback = 20;
C.ml.svmlearner = 'default';
C.ml.cvMode = 'LOO';
C.ml.corr_option = 1;
C.ml.enet_alpha_grid =  [0.5];
C.ml.enet_kfold    = 5;
C.ml.enet_lambda_grid = []; 
C.ml.enet_repeat=10;

%   corr_option : 1 Pearson corr
%               2 Spearman corr
%               3 Robust regression (edge-wise)
%               4 Partial correlation (need cov)
%% ===== GLM参数 =====
C.glm.totalDuration    = 600;     % seconds
C.glm.numPoints        = 290;     % number of TRs
C.glm.highResFactor    = 100;     % 0.01s resolution
C.glm.pulseWidth       = 1;     % seconds
C.glm.dropTR           = 10;      % 你原来用 design_matrix(11:end,:) => drop first 10 TRs

%% ===== Jansen-Rit 模型参数 =====
C.jr.dt        = 1e-3;
C.jr.teq       = 60;
C.jr.tmax      = 660;
C.jr.downsamp  = 1;
C.jr.seed      = 0;
C.jr.verbose   = false;
C.jr.returnBurn = false;

% --- Node dynamics ---
C.jr.a_vel  = 100;
C.jr.ad_vel = 50;
C.jr.b_vel  = 50;

C.jr.p      = 2;
C.jr.sigma  = 2;

C.jr.C  = 135;
C.jr.C1 = 1;
C.jr.C2 = 0.8;
C.jr.C3 = 0.25;
C.jr.C4 = 0.25;

C.jr.A      = 3.25;   % EPSP amplitude
C.jr.B      = 22;     % IPSP amplitude
C.jr.alpha  = 0.5;
C.jr.beta  = 0.25;
% --- Sigmoid ---
C.jr.e0 = 2.5;
C.jr.v0 = 6;
C.jr.r0 = 0.56;
C.jr.r1 = 0.56;
C.jr.r2 = 0.56;

C.jr.stim = 0.5;
C.jr.G = 0.5; 

%% ===== 统计分析参数 =====
C.stats.alpha   = 0.05;
C.stats.n_perm  = 5000;
C.stats.multicorrect = 'BH';


%% ===== 输出与调试 =====
C.output.save_cache        = true;
C.output.overwrite_cache   = false;
C.output.verbose           = true;

end
