function roi2cifti_glasser(roi_values, output_dir, output_name)
%WRITE_ROI_SCALAR_MAP_CIFTI Save 360 ROI values as a CIFTI scalar map.
%
% Assumptions:
%   - roi_values contains 360 parcel values.
%   - Values 1:180 correspond to left hemisphere parcels.
%   - Values 181:360 correspond to right hemisphere parcels.

template_path_left = 'C:\Users\35349\OneDrive\Code\Common\Mask\Glasser_Template\Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii';
template_path_right = 'C:\Users\35349\OneDrive\Code\Common\Mask\Glasser_Template\Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii';
cifti_template_path = 'C:\Users\35349\OneDrive\Code\Common\Mask\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii';

glasser_left = ciftiopen(template_path_left);
glasser_right = ciftiopen(template_path_right);
cifti_scalar = ciftiopen(cifti_template_path);

roi_values = squeeze(roi_values);
roi_values = roi_values(:);

if numel(roi_values) ~= 360
    error('write_roi_scalar_map_cifti:InvalidInput', ...
        'roi_values must contain 360 ROI values, but got %d.', numel(roi_values));
end

for roi_idx = 1:numel(roi_values)
    if roi_idx <= 180
        glasser_left.cdata(glasser_left.cdata == roi_idx) = roi_values(roi_idx);
    else
        right_roi_idx = roi_idx - 180;
        glasser_right.cdata(glasser_right.cdata == right_roi_idx) = roi_values(roi_idx);
    end
end

cifti_scalar.cdata = [glasser_left.cdata; glasser_right.cdata];
cifti_scalar.diminfo{1,2}.type = 'scalars';

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

output_file = fullfile(output_dir, [output_name, '.dscalar.nii']);
ciftisave(cifti_scalar, output_file);
end
