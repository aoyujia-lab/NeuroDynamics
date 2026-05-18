function [R2_quad, p_model, p_quad, mdl] = quadratic_r2(x, y)
%QUADRATIC_FITLM Fit y = b0 + b1*x + b2*x^2 using MATLAB fitlm.
%
% INPUT:
%   x : predictor vector
%   y : response vector
%
% OUTPUT:
%   R2_quad : ordinary R-squared
%   p_model : overall model p-value
%   p_quad  : p-value for quadratic term
%   mdl     : MATLAB LinearModel object

x = x(:);
y = y(:);

valid_idx = ~isnan(x) & ~isnan(y);
x = x(valid_idx);
y = y(valid_idx);

R2_quad = NaN;
p_model = NaN;
p_quad = NaN;
mdl = [];

if numel(y) < 4
    warning('Need at least 4 valid observations.');
    return;
end

if std(x) <= eps || std(y) <= eps
    warning('x or y has near-zero variance.');
    return;
end

% Standardize x before constructing the quadratic term
x_z = zscore(x);
x2 = x_z.^2;

tbl = table(x_z, x2, y, 'VariableNames', {'x', 'x2', 'y'});

mdl = fitlm(tbl, 'y ~ x + x2');

R2_quad = mdl.Rsquared.Ordinary;
p_model = coefTest(mdl);

row_names = mdl.Coefficients.Properties.RowNames;
p_quad = mdl.Coefficients.pValue(strcmp(row_names, 'x2'));

end