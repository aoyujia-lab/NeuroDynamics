function design_matrix = build_design_from_xls(path, total_duration, TR, nTR, high_res_factor)
%BUILD_DESIGN_FROM_XLS Build an event design matrix from the exported XLS file.

if nargin < 5 || isempty(high_res_factor)
    high_res_factor = 100;
end
if nargin < 4 || isempty(nTR)
    nTR = 300;
end
if nargin < 3 || isempty(TR)
    TR = 2;
end
if nargin < 2 || isempty(total_duration)
    total_duration = 600;
end

dt = 1 / high_res_factor;
t_highres = 0:dt:total_duration;
data = xlsread(path);

event_time = (data(:, 1) - data(1, 1)) / 1000;
left_onsets = round(event_time(data(:, 2) == 6), 1);
right_onsets = round(event_time(data(:, 2) == 9) + 2, 1);

pulse_left = build_pulse_train(left_onsets, t_highres, 0.1);
pulse_right = build_pulse_train(right_onsets, t_highres, 0.1);

hrf = spm_hrf(dt);
hrf = hrf(:) ./ max(hrf);
conv_left = conv(pulse_left, hrf, 'full');
conv_right = conv(pulse_right, hrf, 'full');
conv_left = conv_left(1:numel(t_highres));
conv_right = conv_right(1:numel(t_highres));

step = TR / dt;
if abs(step - round(step)) > 1e-9
    error('TR / dt must be an integer.');
end
step = round(step);

reg_left = conv_left(1:step:end);
reg_right = conv_right(1:step:end);
design_matrix = [reg_left(1:nTR).', reg_right(1:nTR).', ones(nTR, 1)];
end

function pulse = build_pulse_train(onsets, t_highres, width)
pulse = zeros(size(t_highres));
for i = 1:numel(onsets)
    pulse((t_highres >= onsets(i)) & (t_highres < onsets(i) + width)) = 1;
end
end
