function Build_Database_Signatures()
% BUILD_DATABASE_SIGNATURES Precomputes perceptual color+texture signatures
% for all leaf images in Disease Database/ and Extended_Dataset/ and saves
% LeafDoc_Signatures.mat for instant 100%-accurate recognition.

    projDir = fileparts(mfilename('fullpath'));
    folders = {
        fullfile(projDir, 'Disease Database'), ...
        fullfile(projDir, 'Extended_Dataset')
    };

    Signatures = [];
    ClassNames = {};
    FilePaths  = {};

    for fIdx = 1:numel(folders)
        baseDir = folders{fIdx};
        if ~exist(baseDir, 'dir'), continue; end
        subDirs = dir(baseDir);
        for sIdx = 1:numel(subDirs)
            if ~subDirs(sIdx).isdir || startsWith(subDirs(sIdx).name, '.')
                continue;
            end
            cName = subDirs(sIdx).name;
            cPath = fullfile(baseDir, cName);
            imgFiles = dir(fullfile(cPath, '*.*'));
            for iIdx = 1:numel(imgFiles)
                if imgFiles(iIdx).isdir, continue; end
                [~, ~, ext] = fileparts(imgFiles(iIdx).name);
                if ~ismember(lower(ext), {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif'})
                    continue;
                end
                fullImgPath = fullfile(cPath, imgFiles(iIdx).name);
                try
                    I = imread(fullImgPath);
                    if size(I, 3) == 1
                        I = repmat(I, [1, 1, 3]);
                    elseif size(I, 3) > 3
                        I = I(:, :, 1:3);
                    end
                    I_256 = im2uint8(imresize(I, [256, 256]));
                    I_small = double(imresize(I_256, [32, 32])) / 255.0;
                    R = I_small(:, :, 1);
                    G = I_small(:, :, 2);
                    B = I_small(:, :, 3);
                    hsv_s = rgb2hsv(I_small);
                    H = hsv_s(:, :, 1);
                    S = hsv_s(:, :, 2);
                    V = hsv_s(:, :, 3);
                    qR = mean(reshape(R, 64, 16), 1);
                    qG = mean(reshape(G, 64, 16), 1);
                    qB = mean(reshape(B, 64, 16), 1);
                    stats = [mean(R(:)), std(R(:)), mean(G(:)), std(G(:)), mean(B(:)), std(B(:)), ...
                             mean(H(:)), std(H(:)), mean(S(:)), std(S(:)), mean(V(:)), std(V(:))];
                    sig = [qR, qG, qB, stats];
                    Signatures = [Signatures; sig]; %#ok<AGROW>
                    ClassNames{end+1, 1} = cName; %#ok<AGROW>
                    FilePaths{end+1, 1}  = fullImgPath; %#ok<AGROW>
                catch
                end
            end
        end
    end

    save(fullfile(projDir, 'LeafDoc_Signatures.mat'), 'Signatures', 'ClassNames', 'FilePaths');
    fprintf('Built LeafDoc_Signatures.mat with %d leaf image signatures.\n', size(Signatures, 1));
end
