function design_matrix = build_design_from_xls(path, totalDuration, samplingInterval, numPoints, highResFactor)
% Minimal function wrapper of your script (smallest changes).
% Output: design_matrix = [s1 s2 intercept] (numPoints x 3)
%
% Requires SPM on path: spm_hrf

% 参数设置（保留你的变量名）
if nargin < 5 || isempty(highResFactor),   highResFactor = 100; end
if nargin < 4 || isempty(numPoints),       numPoints = 300; end
if nargin < 3 || isempty(samplingInterval),samplingInterval = 2; end
if nargin < 2 || isempty(totalDuration),   totalDuration = 600; end

%% 步骤1: 创建高分辨率时间轴
t_highres = linspace(0, totalDuration, totalDuration*highResFactor + 1);

%% 步骤2: 生成高分辨率脉冲序列
data = xlsread(path);

L_pos = find(data(:,2) == 6);
data_L = data(L_pos,1);
R_pos = find(data(:,2) == 9);
data_R = data(R_pos,1);

% 序列1
lag = 0;

data_L = data_L - data(1);
data_L = data_L./1000;
data_L  = data_L + lag;
data_L = round(data_L,1);

pulse1 = zeros(size(t_highres));
for i = 1:length(data_L)
    pt = data_L(i);
    pulse1((t_highres >= pt) & (t_highres < pt + 0.1)) = 1;
end

% 序列2
data_R = data_R - data(1);
data_R = data_R./1000;
data_R  = data_R + lag+2;
data_R = round(data_R,1);

pulse2 = zeros(size(t_highres));
for i = 1:length(data_R)
    pt = data_R(i);
    pulse2((t_highres >= pt) & (t_highres < pt + 0.1)) = 1;
end

%% 步骤3: 使用hrf函数生成HRF核
TR_highres = 1/highResFactor;        % 高分辨率TR（0.01秒）
hrf = spm_hrf(TR_highres);           % 生成HRF核
hrf = hrf/max(hrf);                  % 归一化处理

%% 步骤4: 执行卷积操作
conv1 = conv(pulse1, hrf, 'full');
conv2 = conv(pulse2, hrf, 'full');

%% 步骤5: 对齐和下采样
conv1 = conv1(1:length(t_highres));
conv2 = conv2(1:length(t_highres));

downsampleFactor = samplingInterval/(1/highResFactor); % e.g. 200
if abs(downsampleFactor - round(downsampleFactor)) > 1e-9
    error('samplingInterval / (1/highResFactor) must be integer. Got %.6f', downsampleFactor);
end
downsampleFactor = round(downsampleFactor);

s1 = conv1(1:downsampleFactor:end);
s2 = conv2(1:downsampleFactor:end);

% 确保输出长度
s1 = s1(1:numPoints);
s2 = s2(1:numPoints);

%% 开始回归：输出设计矩阵
design_matrix = [s1(:), s2(:), ones(numPoints,1)];
end