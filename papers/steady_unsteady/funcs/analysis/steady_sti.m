function x = steady_sti(tspan,fs,pulse_duration,pulse_interval,intensity)

t = 0:1/fs:tspan;  % 时间向量
x = zeros(size(t));  % 初始化时间序列，全为 0
for start_time = 0:pulse_interval:(tspan - pulse_duration)
    idx = (t >= start_time) & (t < start_time + pulse_duration);
    x(idx) = intensity;
end

