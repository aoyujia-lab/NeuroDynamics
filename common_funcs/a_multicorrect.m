function out = a_multicorrect(X, P, C)
% A_MULTICORRECT Multiple-comparison correction + apply significance mask.
%
% out = a_multicorrect(X, P, C)
%
% INPUT
%   X : effect / stats map (any shape)
%   P : p-values (same shape as X), MUST be two-sided p-values
%   C : config struct with fields:
%         C.stats.multicorrect  (e.g., 'BH','BY','BONF','HOLM','HOCH','SIDAK','NOC')
%         C.stats.alpha         (e.g., 0.05)
%       Optional:
%         C.stats.mc_dim        for 2D P only:
%              'ALL' (default) : correct across all entries together
%              1 or 'ROW'      : correct within each row separately
%              2 or 'COL'      : correct within each column separately
%
% OUTPUT (struct out)
%   out.X_correct : X with non-significant entries set to 0
%   out.P_correct : corrected p-values (same shape as P)
%   out.mask      : significant mask after correction
%   out.method    : method used
%   out.alpha     : alpha used
%   out.scope     : correction scope ('ALL'/'ROW'/'COL' for 2D; otherwise 'ALL')

% -------------------- config --------------------
method = upper(string(C.stats.multicorrect));
alpha  = C.stats.alpha;

mc_dim = 'ALL';
if isfield(C,'stats') && isfield(C.stats,'mc_dim') && ~isempty(C.stats.mc_dim)
    mc_dim = C.stats.mc_dim;
end

% normalize mc_dim
if isnumeric(mc_dim)
    if mc_dim == 1, mc_dim = 'ROW';
    elseif mc_dim == 2, mc_dim = 'COL';
    else, mc_dim = 'ALL';
    end
else
    mc_dim = upper(string(mc_dim));
    if mc_dim == "ROWS", mc_dim = "ROW"; end
    if mc_dim == "COLS", mc_dim = "COL"; end
    if ~ismember(mc_dim, ["ALL","ROW","COL"])
        mc_dim = "ALL";
    end
end

% -------------------- checks --------------------
assert(isequal(size(X), size(P)), 'X and P must have the same size.');
assert(isnumeric(alpha) && isscalar(alpha) && alpha > 0 && alpha <= 1, ...
    'C.stats.alpha must be a scalar in (0, 1].');

P_correct = nan(size(P));
mask      = false(size(P));

% Optional sanity check: p-values in [0,1] (only for finite)
pvec_all = P(isfinite(P));
if any(pvec_all < 0 | pvec_all > 1)
    warning('Some p-values are outside [0,1]. Please check P.');
end

% -------------------- apply correction --------------------
if ismatrix(P) && ndims(P) == 2
    % 2D special handling
    switch mc_dim
        case "ROW"
            scope = 'ROW';
            for i = 1:size(P,1)
                valid = isfinite(P(i,:));
                pvec  = P(i,valid);
                if isempty(pvec), continue; end
                pc = local_correct_vec(pvec, method);
                P_correct(i,valid) = pc;
                mask(i,valid)      = pc <= alpha;
            end

        case "COL"
            scope = 'COL';
            for j = 1:size(P,2)
                valid = isfinite(P(:,j));
                pvec  = P(valid,j);
                if isempty(pvec), continue; end
                pc = local_correct_vec(pvec, method);
                P_correct(valid,j) = pc;
                mask(valid,j)      = pc <= alpha;
            end

        otherwise % "ALL"
            scope = 'ALL';
            valid = isfinite(P);
            pvec  = P(valid);
            if ~isempty(pvec)
                pc = local_correct_vec(pvec, method);
                P_correct(valid) = pc;
                mask(valid)      = pc <= alpha;
            end
    end

else
    % ND or vector: behave like original (correct across all entries)
    scope = 'ALL';
    valid = isfinite(P);
    pvec  = P(valid);
    if ~isempty(pvec)
        pc = local_correct_vec(pvec, method);
        P_correct(valid) = pc;
        mask(valid)      = pc <= alpha;
    end
end

X_correct = X;
X_correct(~mask) = 0;

out = struct('X_correct', X_correct, ...
             'P_correct', P_correct, ...
             'mask',      mask, ...
             'method',    method, ...
             'alpha',     alpha, ...
             'scope',     scope);
end

% ===================== internal: vector correction =====================
function pc = local_correct_vec(pvec, method)
pvec = pvec(:);
m    = numel(pvec);

switch method
    case {"NOC","NONE"}
        pc = pvec;

    case {"BH","FDR"}
        if exist('mafdr','file') == 2
            pc = mafdr(pvec, 'BHFDR', true);
        else
            pc = fdr_bh_padj(pvec);
        end

    case {"BY"}
        pc = fdr_by_padj(pvec);

    case {"BONF","BONFERRONI","FWE"}
        pc = min(pvec * m, 1);

    case {"SIDAK"}
        pc = 1 - (1 - pvec).^m;

    case {"HOLM"}
        pc = holm_padj(pvec);

    case {"HOCH","HOCHBERG"}
        pc = hochberg_padj(pvec);

    otherwise
        error('Unknown method: %s', method);
end
end

% ===================== helpers =====================
function padj = fdr_bh_padj(p)
p = p(:);
m = numel(p);
[ps, idx] = sort(p, 'ascend');
q = ps .* m ./ (1:m)';          % raw
q = flipud(cummin(flipud(q)));  % enforce monotone
q = min(q, 1);
padj = nan(m,1); padj(idx) = q;
end

function padj = fdr_by_padj(p)
p = p(:);
m = numel(p);
c_m = sum(1./(1:m));
[ps, idx] = sort(p, 'ascend');
q = ps .* m .* c_m ./ (1:m)';
q = flipud(cummin(flipud(q)));
q = min(q, 1);
padj = nan(m,1); padj(idx) = q;
end

function padj = holm_padj(p)
p = p(:);
m = numel(p);
[ps, idx] = sort(p, 'ascend');
adj = (m - (1:m)' + 1) .* ps;
adj = cummax(adj);      % step-down monotonicity
adj = min(adj, 1);
padj = nan(m,1); padj(idx) = adj;
end

function padj = hochberg_padj(p)
p = p(:);
m = numel(p);
[ps, idx] = sort(p, 'ascend');
adj = nan(m,1);
for i = 1:m
    adj(i) = min((m - (i:m)' + 1) .* ps(i:m));
end
adj = min(adj, 1);
adj = cummin(adj(end:-1:1));
adj = adj(end:-1:1);
padj = nan(m,1); padj(idx) = adj;
end