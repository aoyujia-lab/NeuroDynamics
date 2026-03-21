function roi2cifti_glasser(roi_values, output_dir, output_name)
%ROI2CIFTI_GLASSER Save 360 ROI values as a CIFTI scalar map.

left_template = 'C:\Users\35349\OneDrive\Code\Common\Mask\Glasser_Template\Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii';
right_template = 'C:\Users\35349\OneDrive\Code\Common\Mask\Glasser_Template\Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii';
scalar_template = 'C:\Users\35349\OneDrive\Code\Common\Mask\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii';

roi_values = roi_values(:);
if numel(roi_values) ~= 360
    error('roi_values must contain 360 entries, got %d.', numel(roi_values));
end

glasser_left = ciftiopen(left_template);
glasser_right = ciftiopen(right_template);
cifti_scalar = ciftiopen(scalar_template);

for roi_idx = 1:180
    glasser_left.cdata(glasser_left.cdata == roi_idx) = roi_values(roi_idx);
    glasser_right.cdata(glasser_right.cdata == roi_idx) = roi_values(roi_idx + 180);
end

cifti_scalar.cdata = [glasser_left.cdata; glasser_right.cdata];
cifti_scalar.diminfo{1, 2}.type = 'scalars';

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

ciftisave(cifti_scalar, fullfile(output_dir, [output_name, '.dscalar.nii']));
end
