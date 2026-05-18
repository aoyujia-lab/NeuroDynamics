function [F_val, eta2, p_val] = variance_partition(psd_all)
% Variance partition across subjects and runs for each freq × ROI
%
% INPUT:
%   psd_all : [nFreq × nROI × nSubj × nRun]
%
% OUTPUT:
%   F_val : [nFreq × nROI] F ratio (between-subject / within-subject)
%   eta2  : [nFreq × nROI] eta-squared (proportion of variance explained by subject)
%   p_val : [nFreq × nROI] p-value of F test

[nFreq, nROI, nSubj, nRun] = size(psd_all);

F_val = nan(nFreq, nROI);
eta2  = nan(nFreq, nROI);
p_val = nan(nFreq, nROI);

df_between = nSubj - 1;
df_within  = nSubj * (nRun - 1);

for f = 1:nFreq
    for r = 1:nROI
        data = squeeze(psd_all(f, r, :, :));   % [nSubj × nRun]
        
        % handle NaN: only use subjects with complete runs
        valid = all(~isnan(data), 2);
        X = data(valid, :);
        nS = size(X, 1);
        if nS < 3
            continue;
        end
        
        grand_mean = mean(X(:));
        subj_mean  = mean(X, 2);   % [nS × 1]
        
        SS_between = nRun * sum((subj_mean - grand_mean).^2);
        SS_within  = sum(sum((X - subj_mean).^2));
        SS_total   = SS_between + SS_within;
        
        MS_between = SS_between / (nS - 1);
        MS_within  = SS_within  / (nS * (nRun - 1));
        
        F_val(f, r) = MS_between / MS_within;
        eta2(f, r)  = SS_between / SS_total;
        p_val(f, r) = 1 - fcdf(F_val(f, r), nS - 1, nS * (nRun - 1));
    end
    if mod(f, 100) == 0
        fprintf('freq %d/%d done\n', f, nFreq);
    end
end

end