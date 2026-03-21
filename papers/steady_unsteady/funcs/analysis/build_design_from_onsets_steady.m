function X = build_design_from_onsets_steady(L_onsets, R_onsets, C)
%BUILD_DESIGN_FROM_ONSETS_STEADY Build steady-task design matrix in TR space.

dt = 1 / C.glm.highResFactor;
t_highres = 0:dt:C.glm.totalDuration;
hrf = spm_hrf(dt);
hrf = hrf(:) ./ max(hrf);

reg_L = pulse_to_hrf(L_onsets, t_highres, C.glm.pulseWidth, hrf);
reg_R = pulse_to_hrf(R_onsets, t_highres, C.glm.pulseWidth, hrf);

step = C.data.TR / dt;
if abs(step - round(step)) > 1e-9
    error('C.data.TR / dt must be an integer.');
end
step = round(step);

nTR_total = round(C.glm.totalDuration / C.data.TR);
reg_L = reg_L(1:step:end);
reg_R = reg_R(1:step:end);
reg_L = reg_L(1:nTR_total);
reg_R = reg_R(1:nTR_total);

keep_idx = (C.glm.dropTR + 1):nTR_total;
X = [reg_L(keep_idx), reg_R(keep_idx), ones(numel(keep_idx), 1)];
end

function y = pulse_to_hrf(onsets, t_highres, width, hrf)
pulse = zeros(size(t_highres));
for i = 1:numel(onsets)
    onset = onsets(i);
    if isnan(onset) || onset < 0
        continue;
    end
    pulse((t_highres >= onset) & (t_highres < onset + width)) = 1;
end

y = conv(pulse(:), hrf, 'full');
y = y(1:numel(t_highres));
end
