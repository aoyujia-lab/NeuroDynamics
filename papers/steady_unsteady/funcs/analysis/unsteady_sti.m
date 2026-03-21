function x = unsteady_sti(total_time, fs, pulse_duration, min_interval, max_interval, intensity, seed)
%UNSTEADY_STI Generate an irregular but reproducible stimulation train.

if nargin < 7 || isempty(seed)
    seed = 1;
end

rng(seed, 'twister');
t = 0:1/fs:total_time;
x = zeros(size(t));

onset = 0;
while onset < (total_time - pulse_duration)
    x((t >= onset) & (t < onset + pulse_duration)) = intensity;
    onset = onset + min_interval + (max_interval - min_interval) * rand;
end
end
