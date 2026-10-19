function classes = svmclassify(svmStruct, sample, varargin)
% SVMCLASSIFY Compatibility wrapper for modern MATLAB (R2018a+) using predict.
% Supports legacy syntax: classes = svmclassify(svmStruct, tst, 'showplot', false)

    sample = double(sample);
    if isstruct(svmStruct) && isfield(svmStruct, 'IsConstant')
        if svmStruct.IsConstant
            classes = repmat(svmStruct.ConstantClass, size(sample, 1), 1);
        else
            classes = predict(svmStruct.Model, sample);
        end
    else
        classes = predict(svmStruct, sample);
    end
end
