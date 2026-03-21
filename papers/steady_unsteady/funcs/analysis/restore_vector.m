function v360 = restore_vector(v358)
% RESTORE_VECTOR Restore a 358x1 vector to 360x1 by inserting zeros
% at positions 120 and 200.
%
% INPUT
%   v358 : 358x1 vector (after removing elements 120 and 200)
%
% OUTPUT
%   v360 : 360x1 vector with zeros at 120 and 200

    if length(v358) ~= 358
        error('Input vector must be 358x1.');
    end

    removed_idx = [120 200];

    v360 = zeros(360,1);
    keep_idx = setdiff(1:360, removed_idx);

    v360(keep_idx) = v358;
end