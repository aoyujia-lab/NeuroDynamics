function reg = make_hrf_regressor(onsets_sec, duration, TR, total_time)
% MAKE_HRF_REGRESSOR  刺激onset卷积HRF，生成TR分辨率的回归器。
%
% 输入:
%   onsets_sec  - 刺激onset时间(秒)，相对于run起始(0 s)
%   duration    - 刺激持续时间(秒)
%   TR          - 扫描间隔(秒)
%   total_time  - 扫描总时长(秒)
%
% 输出:
%   reg         - 列向量，长度 = total_time / TR

% 高分辨率时间轴 (dt = TR/16)
dt = TR / 16;
t  = 0:dt:total_time;

% 构造刺激方波
pulse = zeros(size(t));
for i = 1:numel(onsets_sec)
    onset = onsets_sec(i);
    if isnan(onset) || onset < 0, continue; end
    pulse(t >= onset & t < onset + duration) = 1;
end

% HRF卷积
hrf = spm_hrf(dt);
hrf = hrf(:) ./ max(hrf);
y   = conv(pulse(:), hrf, 'full');
y   = y(1:numel(t));

% 降采样到TR
step = round(TR / dt);
nTR  = round(total_time / TR);
reg  = y(1:step:end);
reg  = reg(1:nTR);
reg  = reg(:);

end