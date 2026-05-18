function out = paired_perm_loss(pred1, pred2, y, nPerm)
% Paired sign-flip permutation test for difference in MSE.
% + Bootstrap 95% CI for each model's MSE and for mean(MSE diff).
%
% pred1, pred2: [n x 1] predictions from two models
% y: [n x 1] ground-truth values
% nPerm: number of iterations, default 10000

if nargin < 4, nPerm = 10000; end

% Force conversion to column vectors
pred1 = pred1(:); pred2 = pred2(:); y = y(:);
n = numel(y);
assert(numel(pred1)==n && numel(pred2)==n, 'Input vector lengths are inconsistent');

% 1. Compute each model's MSE
se1 = (pred1 - y).^2;
se2 = (pred2 - y).^2;
mse1 = mean(se1);
mse2 = mean(se2);

% 2. Compute observed difference (Observed Statistic)
d = se1 - se2; % Subject-level error difference
T0 = mean(d);  % Equals mse1 - mse2

% 3. Permutation test - sign flip
Tperm = zeros(nPerm, 1);
for b = 1:nPerm
    s = (rand(n, 1) > 0.5) * 2 - 1; 
    Tperm(b) = mean(s .* d);
end

% 4. Compute two-sided p-value
p = (sum(abs(Tperm) >= abs(T0)) + 1) / (nPerm + 1);

% 5. Bootstrap 95% CI（paired resampling）
nBoot = 10000;

boot_mse1 = zeros(nBoot, 1);
boot_mse2 = zeros(nBoot, 1);
boot_diff = zeros(nBoot, 1);

for b = 1:nBoot
    idx = randi(n, n, 1);          % bootstrap indices (paired)
    boot_mse1(b) = mean(se1(idx));
    boot_mse2(b) = mean(se2(idx));
    boot_diff(b) = mean(d(idx));   % = boot_mse1 - boot_mse2 (using d directly is cleaner)
end

ci_mse1 = prctile(boot_mse1, [2.5 97.5]);
ci_mse2 = prctile(boot_mse2, [2.5 97.5]);
ci_meanDiff = prctile(boot_diff, [2.5 97.5]);

% Fill output struct
out.mse1 = mse1;
out.mse2 = mse2;

out.ci_mse1 = ci_mse1;          % [lower upper]
out.ci_mse2 = ci_mse2;          % [lower upper]

out.meanDiff = T0;              % mse1 - mse2
out.ci_meanDiff = ci_meanDiff;  % [lower upper]

out.p = p;
out.nPerm = nPerm;
out.nBoot = nBoot;

end