function [p, diff_obs, diff_perm] = paired_permutation_test(x, y, n_perm, tail)
% permutation_test_paired - paired permutation test (sign-flip)
%
% INPUT:
%   x, y    : paired vectors (same length)
%   n_perm  : number of permutations (e.g., 5000-10000)
%   tail    : 'two-sided' | 'right' | 'left'
%
% OUTPUT:
%   p        : p-value
%   diff_obs : observed mean difference
%   diff_perm: permutation distribution

if nargin < 3 || isempty(n_perm)
    n_perm = 10000;
end
if nargin < 4
    tail = 'two-sided';
end

x = x(:);
y = y(:);

assert(numel(x) == numel(y), 'x and y must have the same length.');

% paired differences
d = x - y;

% observed statistic
diff_obs = mean(d);

n = numel(d);
diff_perm = zeros(n_perm,1);

for i = 1:n_perm
    sign_flip = sign(randn(n,1));   % random +1 / -1
    d_perm = d .* sign_flip;
    diff_perm(i) = mean(d_perm);
end

switch tail
    case 'two-sided'
        p = mean(abs(diff_perm) >= abs(diff_obs));
    case 'right'
        p = mean(diff_perm >= diff_obs);
    case 'left'
        p = mean(diff_perm <= diff_obs);
    otherwise
        error('tail must be two-sided, right, or left');
end
end
