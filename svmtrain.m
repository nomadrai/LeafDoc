function svmStruct = svmtrain(trainData, group, varargin)
% SVMTRAIN Compatibility wrapper for modern MATLAB (R2018a+) using fitcsvm.
% Supports legacy syntax: svmStruct = svmtrain(T, newClass, 'kernel_function', 'linear', ...)

    kernel = 'linear';
    for k = 1:2:length(varargin)
        if k+1 <= length(varargin) && ischar(varargin{k})
            if strcmpi(varargin{k}, 'kernel_function')
                kernel = lower(varargin{k+1});
            end
        end
    end
    if strcmp(kernel, 'rbf')
        kernel = 'gaussian';
    end

    group = group(:);
    trainData = double(trainData);

    % Handle degenerate single-class edge case gracefully
    u = unique(group);
    if numel(u) < 2
        svmStruct = struct('IsConstant', true, 'ConstantClass', u(1), 'Model', []);
        return;
    end

    try
        mdl = fitcsvm(trainData, group, ...
            'KernelFunction', kernel, ...
            'Standardize', true, ...
            'KernelScale', 'auto');
        svmStruct = struct('IsConstant', false, 'ConstantClass', [], 'Model', mdl);
    catch
        mdl = fitcsvm(trainData, group, 'KernelFunction', 'linear');
        svmStruct = struct('IsConstant', false, 'ConstantClass', [], 'Model', mdl);
    end
end
