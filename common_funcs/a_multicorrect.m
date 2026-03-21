function out = a_multicorrect(X, P, C)
%A_MULTICORRECT Multiple-comparison correction with significance masking.

method = upper(string(C.stats.multicorrect));
alpha = C.stats.alpha;
scope = normalize_scope(C);

assert(isequal(size(X), size(P)), 'X and P must have the same size.');
assert(isnumeric(alpha) && isscalar(alpha) && alpha > 0 && alpha <= 1, ...
    'C.stats.alpha must be a scalar in (0, 1].');

P_correct = nan(size(P));
mask = false(size(P));

if ismatrix(P) && ndims(P) == 2 && scope ~= "ALL"
    switch scope
        case "ROW"
            for i = 1:size(P, 1)
                [P_correct(i, :), mask(i, :)] = correct_slice(P(i, :), method, alpha);
            end
        case "COL"
            for j = 1:size(P, 2)
                [P_correct(:, j), mask(:, j)] = correct_slice(P(:, j), method, alpha);
            end
    end
else
    [P_correct, mask] = correct_slice(P, method, alpha);
    scope = "ALL";
end

X_correct = X;
X_correct(~mask) = 0;

out = struct( ...
    'X_correct', X_correct, ...
    'P_correct', P_correct, ...
    'mask', mask, ...
    'method', method, ...
    'alpha', alpha, ...
    'scope', char(scope));
end

function scope = normalize_scope(C)
scope = "ALL";
if ~isfield(C, 'stats') || ~isfield(C.stats, 'mc_dim') || isempty(C.stats.mc_dim)
    return;
end

mc_dim = C.stats.mc_dim;
if isnumeric(mc_dim)
    if mc_dim == 1
        scope = "ROW";
    elseif mc_dim == 2
        scope = "COL";
    end
    return;
end

switch upper(string(mc_dim))
    case {"ROW", "ROWS"}
        scope = "ROW";
    case {"COL", "COLS"}
        scope = "COL";
end
end

function [p_correct, mask] = correct_slice(p_in, method, alpha)
p_correct = nan(size(p_in));
mask = false(size(p_in));
valid = isfinite(p_in);
if ~any(valid)
    return;
end

p_vec = p_in(valid);
p_correct(valid) = local_correct_vec(p_vec, method);
mask(valid) = p_correct(valid) <= alpha;
end

function pc = local_correct_vec(pvec, method)
pvec = pvec(:);
m = numel(pvec);

switch method
    case {"NOC", "NONE"}
        pc = pvec;
    case {"BH", "FDR"}
        if exist('mafdr', 'file') == 2
            pc = mafdr(pvec, 'BHFDR', true);
        else
            pc = fdr_bh_padj(pvec);
        end
    case "BY"
        pc = fdr_by_padj(pvec);
    case {"BONF", "BONFERRONI", "FWE"}
        pc = min(pvec * m, 1);
    case "SIDAK"
        pc = 1 - (1 - pvec) .^ m;
    case "HOLM"
        pc = holm_padj(pvec);
    case {"HOCH", "HOCHBERG"}
        pc = hochberg_padj(pvec);
    otherwise
        error('Unknown method: %s', method);
end
end

function padj = fdr_bh_padj(p)
p = p(:);
m = numel(p);
[ps, idx] = sort(p, 'ascend');
q = ps .* m ./ (1:m)';
q = flipud(cummin(flipud(q)));
padj = nan(m, 1);
padj(idx) = min(q, 1);
end

function padj = fdr_by_padj(p)
p = p(:);
m = numel(p);
[ps, idx] = sort(p, 'ascend');
q = ps .* m .* sum(1 ./ (1:m)) ./ (1:m)';
q = flipud(cummin(flipud(q)));
padj = nan(m, 1);
padj(idx) = min(q, 1);
end

function padj = holm_padj(p)
p = p(:);
m = numel(p);
[ps, idx] = sort(p, 'ascend');
adj = cummax((m - (1:m)' + 1) .* ps);
padj = nan(m, 1);
padj(idx) = min(adj, 1);
end

function padj = hochberg_padj(p)
p = p(:);
m = numel(p);
[ps, idx] = sort(p, 'ascend');
adj = nan(m, 1);
for i = 1:m
    adj(i) = min((m - (i:m)' + 1) .* ps(i:m));
end
adj = adj(end:-1:1);
adj = cummin(adj);
adj = adj(end:-1:1);
padj = nan(m, 1);
padj(idx) = min(adj, 1);
end
