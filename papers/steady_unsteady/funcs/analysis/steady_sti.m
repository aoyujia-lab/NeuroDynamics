function x = steady_sti(total_time, fs, pulse_duration, pulse_interval, intensity)
%STEADY_STI Generate a periodic stimulation train.

t = 0:1/fs:total_time;
x = zeros(size(t));

for onset = 0:pulse_interval:(total_time - pulse_duration)
    x((t >= onset) & (t < onset + pulse_duration)) = intensity;
end
end
