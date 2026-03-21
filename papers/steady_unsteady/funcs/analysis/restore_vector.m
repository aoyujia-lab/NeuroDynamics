function v_full = restore_vector(v_keep, C)
%RESTORE_VECTOR Restore a vector after removing excluded ROIs.

removed_idx = C.data.excluderoi(:)';
nROI = C.data.nROI;
keep_idx = setdiff(1:nROI, removed_idx, 'stable');

if numel(v_keep) ~= numel(keep_idx)
    error('Input vector length (%d) does not match the number of kept ROIs (%d).', ...
        numel(v_keep), numel(keep_idx));
end

v_full = zeros(nROI, 1);
v_full(keep_idx) = v_keep(:);
end
