function [feat_disease, seg_img, Affect, roi_cluster, segmented_images, info] = EvaluateFeatures(I, auto_select, chosen_cluster)
% EVALUATEFEATURES Segment leaf image via K-Means in L*a*b* space & extract 13 GLCM/statistical features.
% Inputs:
%   I              - Input RGB leaf image (resized to 256x256 if needed)
%   auto_select    - (Optional) true = automatically pick disease ROI cluster (default: true)
%                    false = show cluster dialog (pre-filled with recommended ROI cluster)
%   chosen_cluster - (Optional) 1, 2, or 3 to force a specific cluster
% Outputs:
%   feat_disease     - 1x13 feature vector [Contrast, Correlation, Energy, Homogeneity, ...
%                      Mean, Std_Dev, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM]
%   seg_img          - Segmented RGB image of the selected ROI cluster
%   Affect           - Percentage of leaf area affected by disease (0 - 100)
%   roi_cluster      - Index (1, 2, or 3) of the disease ROI cluster
%   segmented_images - 1x3 cell array of the 3 K-Means cluster images
%   info             - Struct with detailed masks, cluster roles, and color metrics

    if nargin < 2 || isempty(auto_select)
        auto_select = true;
    end
    if nargin < 3
        chosen_cluster = [];
    end

    % Ensure 3-channel uint8 RGB image of size 256x256
    if size(I, 3) == 1
        I = repmat(I, [1, 1, 3]);
    elseif size(I, 3) > 3
        I = I(:, :, 1:3);
    end
    I = im2uint8(imresize(I, [256, 256]));

    % Convert from RGB to CIE L*a*b* color space
    try
        lab_he = rgb2lab(I);
    catch
        cform = makecform('srgb2lab');
        lab_he = double(applycform(I, cform));
    end

    ab = double(lab_he(:, :, 2:3));
    nrows = size(ab, 1);
    ncols = size(ab, 2);
    ab_flat = reshape(ab, nrows * ncols, 2);
    nColors = 3;

    % Deterministic initialization using quantiles of a* (green-red axis)
    % so K-Means produces consistent, reproducible clusters every single time
    a_vals = ab_flat(:, 1);
    b_vals = ab_flat(:, 2);
    [~, sortIdx] = sort(a_vals);
    nPix = numel(a_vals);
    init_centers = [
        mean(ab_flat(sortIdx(1:round(0.33*nPix)), :), 1);
        mean(ab_flat(sortIdx(round(0.33*nPix)+1:round(0.66*nPix)), :), 1);
        mean(ab_flat(sortIdx(round(0.66*nPix)+1:end), :), 1)
    ];

    try
        [cluster_idx, ~] = kmeans(ab_flat, nColors, ...
            'Distance', 'sqEuclidean', ...
            'Start', init_centers, ...
            'MaxIter', 150);
    catch
        rng(42);
        [cluster_idx, ~] = kmeans(ab_flat, nColors, ...
            'Distance', 'sqEuclidean', ...
            'Replicates', 3);
    end

    pixel_labels = reshape(cluster_idx, nrows, ncols);

    % Analyze each cluster to identify:
    %   1) Background cluster (touches image border heavily & low greenness / extreme brightness)
    %   2) Healthy Green Leaf cluster (most negative a* and high Excess Green 2G-R-B)
    %   3) Disease Lesion ROI cluster (lesions/spots: higher a*, brown/yellow/dark/white spots)
    R = double(I(:, :, 1)) / 255.0;
    G = double(I(:, :, 2)) / 255.0;
    B = double(I(:, :, 3)) / 255.0;
    L_chan = double(lab_he(:, :, 1));
    a_chan = double(lab_he(:, :, 2));
    b_chan = double(lab_he(:, :, 3));
    ExG = 2 * G - R - B;

    border_mask = false(nrows, ncols);
    border_mask([1:6, end-5:end], :) = true;
    border_mask(:, [1:6, end-5:end]) = true;

    bg_score = zeros(1, nColors);
    green_score = zeros(1, nColors);
    lesion_score = zeros(1, nColors);
    cluster_area = zeros(1, nColors);

    for k = 1:nColors
        mask_k = (pixel_labels == k);
        cluster_area(k) = sum(mask_k(:));
        if cluster_area(k) == 0
            bg_score(k) = 999;
            continue;
        end
        border_frac = sum(mask_k(border_mask)) / sum(border_mask(:));
        mean_ExG = mean(ExG(mask_k));
        mean_L = mean(L_chan(mask_k));
        mean_a = mean(a_chan(mask_k));
        mean_b = mean(b_chan(mask_k));
        sat_val = mean(max(cat(3, R(mask_k), G(mask_k), B(mask_k)), [], 3) - ...
                       min(cat(3, R(mask_k), G(mask_k), B(mask_k)), [], 3));

        % Background is typically on the border with low saturation or very high/low L and low ExG
        bg_score(k) = border_frac * 2.5 - mean_ExG * 1.5 + (sat_val < 0.12) * 1.0 + ...
                      ((mean_L > 85) || (mean_L < 12)) * 0.8;

        % Healthy leaf tissue has strong Excess Green and negative a*
        green_score(k) = mean_ExG * 3.0 - (mean_a / 20.0) + (mean_b / 40.0);

        % Lesion/Disease ROI has higher a* (red/brown/tan) or yellow/dark discoloration
        lesion_score(k) = (mean_a / 15.0) - mean_ExG * 1.5 + abs(mean_b) / 35.0;
    end

    [~, bg_cluster] = max(bg_score);
    rem_clusters = setdiff(1:nColors, bg_cluster);
    if green_score(rem_clusters(1)) >= green_score(rem_clusters(2))
        healthy_cluster = rem_clusters(1);
        disease_cluster = rem_clusters(2);
    else
        healthy_cluster = rem_clusters(2);
        disease_cluster = rem_clusters(1);
    end

    % Reorder clusters deterministically so:
    %   Cluster 1 = Disease ROI Cluster
    %   Cluster 2 = Healthy Leaf Tissue Cluster
    %   Cluster 3 = Background / Outer Cluster
    new_order = [disease_cluster, healthy_cluster, bg_cluster];
    reordered_labels = zeros(size(pixel_labels));
    for k = 1:nColors
        reordered_labels(pixel_labels == new_order(k)) = k;
    end
    pixel_labels = reordered_labels;

    segmented_images = cell(1, nColors);
    rgb_label = repmat(pixel_labels, [1, 1, 3]);
    for k = 1:nColors
        colors = I;
        colors(rgb_label ~= k) = 0;
        segmented_images{k} = colors;
    end

    roi_cluster = 1; % Since we ordered Cluster 1 = Disease ROI!

    if ~isempty(chosen_cluster) && chosen_cluster >= 1 && chosen_cluster <= 3
        roi_cluster = chosen_cluster;
    elseif ~auto_select
        % Show the 3 clusters and pre-fill '1' (Disease ROI) so user can just click OK
        hFig = figure('Name', 'Segmented by K-Means (Cluster 1 = Recommended Disease ROI)', ...
                      'NumberTitle', 'off', 'Color', [0.95 0.97 0.95]);
        subplot(2, 2, 1); imshow(I); title('Original Enhanced Image', 'FontWeight', 'bold');
        subplot(2, 2, 2); imshow(segmented_images{1}); title('Cluster 1: Disease ROI (Recommended)', 'Color', [0.8 0.1 0.1], 'FontWeight', 'bold');
        subplot(2, 2, 3); imshow(segmented_images{2}); title('Cluster 2: Healthy Green Tissue', 'Color', [0.1 0.5 0.1]);
        subplot(2, 2, 4); imshow(segmented_images{3}); title('Cluster 3: Background / Border');
        drawnow;
        x = inputdlg({'Enter the cluster no. containing the Disease ROI (Default = 1):'}, ...
                     'Select K-Means ROI Cluster', [1 50], {'1'});
        if ishandle(hFig)
            close(hFig);
        end
        if ~isempty(x)
            val = str2double(x{1});
            if ~isnan(val) && val >= 1 && val <= 3
                roi_cluster = round(val);
            end
        end
    end

    seg_img = segmented_images{roi_cluster};

    % Convert to grayscale
    if ndims(seg_img) == 3
        img = rgb2gray(seg_img);
    else
        img = seg_img;
    end

    % Compute accurate Leaf Mask and Disease Lesion Mask for Affected Area (%)
    leaf_mask = (pixel_labels == 1) | (pixel_labels == 2);
    % Refine leaf mask if background was misassigned on full-frame leaf photos
    if sum(leaf_mask(:)) < 0.15 * (nrows * ncols)
        leaf_mask = true(nrows, ncols);
    end
    leaf_mask = bwareaopen(leaf_mask, 30);
    total_leaf_pixels = max(1, sum(leaf_mask(:)));

    % Detect true lesion pixels within the leaf mask using chromatic + intensity deviation
    roi_mask = (pixel_labels == roi_cluster) & leaf_mask;
    % Also measure how "healthy green" the ROI cluster actually is:
    % If the entire leaf is uniformly healthy green, the ROI cluster is just a shade of green!
    roi_ExG = mean(ExG(roi_mask));
    roi_a = mean(a_chan(roi_mask));
    leaf_ExG = mean(ExG(leaf_mask));

    % True necrotic/chlorotic/powdery/rust lesion mask inside the leaf
    lesion_mask = leaf_mask & ( ...
        (a_chan > -2.5) | ...                                % Brown / red / tan / necrotic / grey
        (ExG < 0.06 & L_chan < 82) | ...                     % Non-green dark or dull spots
        (b_chan > 28 & a_chan > -8) | ...                    % Strong yellow chlorosis / rust
        (L_chan > 75 & abs(a_chan) < 6 & abs(b_chan) < 10) ... % White powdery mildew spots
    );
    lesion_mask = bwareaopen(lesion_mask, 8);

    raw_lesion_ratio = sum(lesion_mask(:)) / total_leaf_pixels;
    cluster_ratio = sum(roi_mask(:)) / total_leaf_pixels;

    if raw_lesion_ratio < 0.025 && leaf_ExG > 0.18 && roi_a < -5.0
        % Healthy leaf
        Affect = raw_lesion_ratio * 100;
    else
        Affect = max(raw_lesion_ratio * 100, min(cluster_ratio * 100, 65.0));
        if Affect < 5.0 && raw_lesion_ratio >= 0.015
            Affect = raw_lesion_ratio * 100 + 8.5;
        end
    end
    Affect = min(95.0, max(0.0, Affect));

    % Extract the 13 standard GLCM + statistical features (matching Training_Data.mat format)
    glcms = graycomatrix(img);
    stats = graycoprops(glcms, 'Contrast Correlation Energy Homogeneity');
    Contrast = stats.Contrast;
    Correlation = stats.Correlation;
    Energy = stats.Energy;
    Homogeneity = stats.Homogeneity;
    if isnan(Correlation)
        Correlation = 0;
    end

    Mean = mean2(seg_img);
    Standard_Deviation = std2(seg_img);
    Entropy = entropy(seg_img);
    RMS = mean2(rms(double(seg_img)));
    Variance = mean2(var(double(seg_img)));
    a_sum = sum(double(seg_img(:)));
    Smoothness = 1 - (1 / (1 + a_sum));
    Kurtosis = kurtosis(double(seg_img(:)));
    Skewness = skewness(double(seg_img(:)));
    if isnan(Kurtosis), Kurtosis = 0; end
    if isnan(Skewness), Skewness = 0; end

    % Fast vectorized Inverse Difference Moment (IDM)
    [rGrid, cGrid] = ndgrid(1:nrows, 1:ncols);
    denom = 1 + (rGrid - cGrid).^2;
    % Compute both true double IDM and legacy uint8-compatible IDM
    true_IDM = sum(sum(double(img) ./ denom));
    IDM = min(255.0, max(0.0, true_IDM));

    feat_disease = [Contrast, Correlation, Energy, Homogeneity, ...
                    Mean, Standard_Deviation, Entropy, RMS, ...
                    Variance, Smoothness, Kurtosis, Skewness, IDM];

    % Populate diagnostic info struct for advanced classification & visualization
    info = struct();
    info.leaf_mask = leaf_mask;
    info.lesion_mask = lesion_mask;
    info.roi_mask = roi_mask;
    info.pixel_labels = pixel_labels;
    info.raw_lesion_ratio = raw_lesion_ratio;
    info.cluster_ratio = cluster_ratio;
    info.leaf_ExG = leaf_ExG;
    info.roi_ExG = roi_ExG;
    info.roi_a = roi_a;
    info.true_IDM = true_IDM;
end
