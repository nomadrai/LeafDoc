% Project Title: Plant Leaf Disease Detection using Image Processing
% Standalone Interactive Script (Detect.m)
% Can be run directly in MATLAB Command Window: >> Detect

clc;
close all;

projDir = fileparts(mfilename('fullpath'));
addpath(projDir);

defaultDir = fullfile(projDir, 'Extended_Dataset');
if ~exist(defaultDir, 'dir')
    defaultDir = fullfile(projDir, 'Disease Database');
end

[filename, pathname] = uigetfile( ...
    {'*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tif', 'Leaf Image Files (*.jpg, *.png, *.bmp, *.gif, *.tif)'; ...
     '*.*', 'All Files (*.*)'}, ...
    'Pick a Leaf Image File', defaultDir);

if isequal(filename, 0) || isequal(pathname, 0)
    disp('User cancelled image selection.');
    return;
end

fullpath = fullfile(pathname, filename);
I_raw = imread(fullpath);
if size(I_raw, 3) == 1
    I_raw = repmat(I_raw, [1, 1, 3]);
elseif size(I_raw, 3) > 3
    I_raw = I_raw(:, :, 1:3);
end
I = im2uint8(imresize(I_raw, [256, 256]));

% Enhance Contrast
I_enh = imadjust(I, stretchlim(I));

% K-Means Segmentation & 13 GLCM/Statistical Feature Extraction
[feat_disease, seg_img, Affect, roi_cluster, segmented_images, info] = EvaluateFeatures(I_enh, true);

% Universal Disease Classification & Symptom Analysis
report = AnalyzeAnyLeaf(I, feat_disease, seg_img, Affect, info, fullpath);

% Display Visual Pipeline Summary Figure
figure('Name', ['LeafDoc Analysis: ', report.DiseaseName], 'NumberTitle', 'off', ...
       'Position', [100, 100, 1100, 650]);
subplot(2, 3, 1); imshow(I); title('1. Query Leaf Image');
subplot(2, 3, 2); imshow(I_enh); title('2. Contrast Enhanced');
subplot(2, 3, 3); imshow(segmented_images{1}); title('3. Cluster 1 (Disease ROI)');
subplot(2, 3, 4); imshow(segmented_images{2}); title('4. Cluster 2 (Healthy Tissue)');
subplot(2, 3, 5); imshow(segmented_images{3}); title('5. Cluster 3 (Background)');
subplot(2, 3, 6); imshow(report.OverlayImage);
title(sprintf('6. Diagnosis: %s (%.1f%%)', report.DiseaseName, report.AffectedArea));

fprintf('\n======================================================\n');
fprintf('         LEAF DISEASE DETECTION REPORT                \n');
fprintf('======================================================\n');
fprintf('Image File       : %s\n', filename);
fprintf('Detected Disease : %s\n', report.DiseaseName);
fprintf('Pathogen Type    : %s\n', report.PathogenType);
fprintf('Confidence       : %.1f%%\n', report.Confidence);
fprintf('Affected Area    : %.2f%% (%s)\n', report.AffectedArea, report.Severity);
fprintf('Symptoms         : %s\n', report.Symptoms);
fprintf('Treatment/Remedy : %s\n', report.Remedy);
fprintf('======================================================\n');

helpdlg(sprintf(['Detected Disease: %s\n' ...
                 'Pathogen Type: %s\n' ...
                 'Affected Area: %.2f%% (%s)\n' ...
                 'Confidence: %.1f%%\n\n' ...
                 'Treatment:\n%s'], ...
                 report.DiseaseName, report.PathogenType, ...
                 report.AffectedArea, report.Severity, ...
                 report.Confidence, report.Remedy), ...
        ['Disease Diagnosis: ', report.DiseaseName]);
