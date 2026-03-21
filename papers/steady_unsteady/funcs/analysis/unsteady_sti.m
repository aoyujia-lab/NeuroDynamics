function x = unsteady_sti(T,fs,pulse_duration,pulse_min_interval,pulse_max_interval,intensity,seed)

if nargin < 7
    seed = 1;   % 默认种子，可自己改
end

rng(seed,'twister');   % 固定随机数 → 伪随机（可复现）

t = 0:1/fs:T;  
x = zeros(size(t));

current_time = 0;

while current_time < (T - pulse_duration)
    idx = (t >= current_time) & (t < current_time + pulse_duration);
    x(idx) = intensity;

    pulse_interval = pulse_min_interval + ...
        (pulse_max_interval - pulse_min_interval) * rand;

    current_time = current_time + pulse_interval;
end