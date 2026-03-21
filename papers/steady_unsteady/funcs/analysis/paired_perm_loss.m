function out = paired_perm_loss(pred1, pred2, y, nPerm)
% Paired sign-flip permutation test for difference in MSE.
% + Bootstrap 95% CI for each model's MSE and for mean(MSE diff).
%
% pred1, pred2: [n x 1] 两个模型的预测值
% y: [n x 1] 真实值
% nPerm: 迭代次数，默认 10000

if nargin < 4, nPerm = 10000; end

% 强制转换为列向量
pred1 = pred1(:); pred2 = pred2(:); y = y(:);
n = numel(y);
assert(numel(pred1)==n && numel(pred2)==n, '输入向量长度不一致');

% 1. 计算各自的 MSE
se1 = (pred1 - y).^2;
se2 = (pred2 - y).^2;
mse1 = mean(se1);
mse2 = mean(se2);

% 2. 计算观测到的差异（Observed Statistic）
d = se1 - se2; % 个体层面的误差差值
T0 = mean(d);  % 即 mse1 - mse2

% 3. 置换检验 (Permutation Test) - sign flip
Tperm = zeros(nPerm, 1);
for b = 1:nPerm
    s = (rand(n, 1) > 0.5) * 2 - 1; 
    Tperm(b) = mean(s .* d);
end

% 4. 计算双侧 P 值
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
    boot_diff(b) = mean(d(idx));   % = boot_mse1 - boot_mse2 (但直接用 d 更干净)
end

ci_mse1 = prctile(boot_mse1, [2.5 97.5]);
ci_mse2 = prctile(boot_mse2, [2.5 97.5]);
ci_meanDiff = prctile(boot_diff, [2.5 97.5]);

% 填充输出结构体
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