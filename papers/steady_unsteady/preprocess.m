clear
subj_path = 'F:\Data\Steady-unsteady\BIDS_derivative';

subj_file = dir(fullfile(subj_path,'sub*'));

wbcmdPath = 'D:\Software\workbench\bin_windows64\wb_command';

load('D:\Code\CBIG-master\stable_projects\brain_parcellation\Kong2019_MSHBM\lib\fs_LR_32k_medial_mask.mat');
atlas_path = 'D:\Code\CBIG-master\stable_projects\brain_parcellation\Schaefer2018_LocalGlobal\Parcellations\HCP\fslr32k\cifti\Schaefer2018_100Parcels_17Networks_order.dlabel.nii';

template_path = 'D:\Code\Common\Mask\Glasser_Template\Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii';
glasser_L = ciftiopen(template_path);
template_path = 'D:\Code\Common\Mask\Glasser_Template\Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii';
glasser_R = ciftiopen(template_path);

label = unique(glasser_L.cdata);

% altlasData = ciftiopen(atlas_path);
% label = 1:100;
% surf_pos = altlasData.cdata;

for isubj = 1:length(subj_file)
    isubj
    ses_file = dir(fullfile(subj_path,subj_file(isubj).name,'ses*'));
    for ises = 1:length(ses_file)
        cifti_file = dir(fullfile(subj_path,subj_file(isubj).name,ses_file(ises).name,'func','*.dtseries.nii'));

        cifti_path = fullfile(subj_path,subj_file(isubj).name,ses_file(ises).name,'func',cifti_file.name);
        ciftiData = ciftiopen(cifti_path,wbcmdPath);

        signals = ciftiData.cdata;
        signals = double(signals(1:59412,:));

        confound_file = dir(fullfile(subj_path,subj_file(isubj).name,ses_file(ises).name,'func','*.tsv'));
        confound_path = fullfile(subj_path,subj_file(isubj).name,ses_file(ises).name,'func',confound_file.name);

        confound = readtable(confound_path, "FileType","text",'Delimiter', '\t');

        csf = confound.csf;
        wm = confound.white_matter;
        FD = confound.framewise_displacement;
        hm1 = confound.trans_x;
        hm2 = confound.trans_y;
        hm3 = confound.trans_z;
        hm4 = confound.rot_x;
        hm5 = confound.rot_y;
        hm6 = confound.rot_z;

        hm1_d = confound.trans_x_derivative1;
        hm2_d = confound.trans_y_derivative1;
        hm3_d = confound.trans_z_derivative1;
        hm4_d = confound.rot_x_derivative1;
        hm5_d = confound.rot_y_derivative1;
        hm6_d = confound.rot_z_derivative1;

        mean_FD(isubj,ises) = mean(FD(2:end));




        x_csf = [csf, ones(size(hm1,1),1)];
        x_wm = [wm, ones(size(hm1,1),1)];
        x_FD = [FD, ones(size(hm1,1),1)];
        x_hm = [hm1,hm2,hm3,hm4,hm5,hm6, ones(size(hm1,1),1)];
        x_hm_d = [hm1_d,hm2_d,hm3_d,hm4_d,hm5_d,hm6_d, ones(size(hm1,1),1)];


        parfor ivoxel = 1:size(signals,1);

            % [b,bint,r,rint,stats] = regress(signals(ivoxel,2:end)',x_csf);
            % [b,bint,r2,rint,stadts] = regress(r,x_wm);
            % [b,bint,r3,rint,stadts] = regress(r2,x_FD);
            % [b,bint,r4,rint,stadts] = regress(r3,x_hm);

            [b,r,SSE,SSR] = y_regress_ss(signals(ivoxel,2:end)',x_csf(2:end,:));
            [b,r2,SSE,SSR] = y_regress_ss(r,x_wm(2:end,:));
            [b,r3,SSE,SSR] = y_regress_ss(r2,x_FD(2:end,:));
            [b,r4,SSE,SSR] = y_regress_ss(r3,x_hm(2:end,:));
            [b,r5,SSE,SSR] = y_regress_ss(r4,x_hm_d(2:end,:));
            out_signals(ivoxel,:) = r5;  
       end

        % % gsr
        GS = mean(out_signals,1)';
        % x_gs = [GS,ones(size(GS,1),1)];
        % parfor ivoxel = 1:size(signals,1);
        %     [b,r6,SSE,SSR] = y_regress_ss(out_signals(ivoxel,:)',x_gs);
        %     out_signals(ivoxel,:) = r6;
        % end

        surf_L = out_signals(1:29696,:);
        surf_R = out_signals(29697:59412,:);


        label = unique(glasser_L.cdata);
        for iroi = 1:length(label)
            roi_pos = find(glasser_L.cdata == label(iroi));
            roi_voxel = surf_L(roi_pos,:);
            roisignals_L(iroi,:) = mean(roi_voxel,1);

            roi_pos = find(glasser_R.cdata == label(iroi));
            roi_voxel = surf_R(roi_pos,:);
            roisignals_R(iroi,:) = mean(roi_voxel,1);
        end

        roisignals = [roisignals_L;roisignals_R];
        roisignals = roisignals(:,10:end);









        % label_0 = find(surf_pos==0);
        %
        %
        % fMRI_data = zeros(64984,size(out_signals,2));
        % fMRI_data(medial_mask,:) = out_signals;
        %
        % % zscore
        % % fMRI_data = zscore(fMRI_data')';
        %
        % fMRI_data_copy = fMRI_data;
        % fMRI_data_copy(label_0,:) = [];
        % GS = mean(fMRI_data_copy,1)';
        % GS = GS(10:end);
        %
        % roisignals = zeros(100,size(out_signals,2));
        %
        % for iroi = 1:length(label)
        %     roi_pos = find(surf_pos == label(iroi));
        %     roi_voxel = fMRI_data(roi_pos,:);
        %     roisignals(iroi,:) = mean(roi_voxel,1);
        % end
        %
        % roisignals = roisignals(:,10:end);


        % roisignals = zscore(roisignals')';

        mkdir(fullfile('E:\Steady-Unsteady\roisignals',subj_file(isubj).name,ses_file(ises).name));
        cd(fullfile('E:\Steady-Unsteady\roisignals',subj_file(isubj).name,ses_file(ises).name));
        save roisignals roisignals GS

    end
end

