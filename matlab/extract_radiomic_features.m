function [features, featureNames, details] = extract_radiomic_features(imgROI)
%EXTRACT_RADIOMIC_FEATURES Extract the 280 radiomic descriptors used by the framework.
%
% INPUT
%   imgROI : 2-D grayscale image ROI.
%
% OUTPUT
%   features     : 1 x 280 double vector
%   featureNames : 1 x 280 string array
%   details      : structure containing extraction diagnostics
%
% The extraction procedure uses:
%   - 56 radiomic features per image domain
%   - original B-mode ROI
%   - level-1 Haar LL, LH, HL, and HH subbands
%   - Fixed Bin Number discretization with 32 bins
%   - no intensity resegmentation
%   - no spatial resampling
%   - signed wavelet coefficients without subband normalization
%
% The five domains therefore yield 56 x 5 = 280 descriptors.

arguments
    imgROI {mustBeNumeric, mustBeNonempty}
end

if ndims(imgROI) ~= 2
    error("imgROI must be a 2-D grayscale matrix.");
end

baseFeatureNames = radiomic_feature_names();
domains = ["O","LL","LH","HL","HH"];

% Original B-mode data are retained on their native grayscale scale.
% Wavelet decomposition is performed after conversion to double precision
% using the fixed native image range.
[LL,LH,HL,HH] = dwt2(im2double(imgROI),"haar");

dataCell = { ...
    imgROI, ...
    double(LL), ...
    double(LH), ...
    double(HL), ...
    double(HH) ...
};

features = nan(1,56*5);
featureNames = strings(1,56*5);

details = struct();
details.FBN = 32;
details.Domains = domains;
details.Rows = zeros(1,5);
details.Cols = zeros(1,5);
details.Min = zeros(1,5);
details.Max = zeros(1,5);
details.Resample = false(1,5);
details.ReturnedOrderMatchedRequested = false(1,5);

for d = 1:5

    data = dataCell{d};
    mask = uint8(ones(size(data)));

    R = radiomics( ...
        data, ...
        mask, ...
        Resegment=false, ...
        Discretize=true, ...
        DiscreteMethod="FixedBinNumber", ...
        DiscreteBinSizeOrBinNumber=32);

    T = selectFeatures(R,baseFeatureNames);

    % MATLAB returns LabelID first, followed by the selected features.
    returnedNames = string(T.Properties.VariableNames(2:end));

    % Validate feature identity independently of returned column order.
    if numel(returnedNames) ~= 56
        error( ...
            "Domain %s: expected 56 feature columns, received %d.", ...
            domains(d), ...
            numel(returnedNames));
    end

    if numel(unique(returnedNames)) ~= 56
        error( ...
            "Domain %s: duplicate feature names returned by selectFeatures.", ...
            domains(d));
    end

    missing = setdiff(baseFeatureNames,returnedNames,"stable");
    unexpected = setdiff(returnedNames,baseFeatureNames,"stable");

    if ~isempty(missing) || ~isempty(unexpected)
        error( ...
            "Domain %s: feature identity mismatch. Missing=[%s], Unexpected=[%s].", ...
            domains(d), ...
            strjoin(missing,", "), ...
            strjoin(unexpected,", "));
    end

    % Explicitly restore the predefined feature order.
    [tf,loc] = ismember(baseFeatureNames,returnedNames);

    if ~all(tf) || any(loc == 0)
        error( ...
            "Domain %s: failed to map returned features to the requested order.", ...
            domains(d));
    end

    vals = table2array(T(:,loc+1));

    if numel(vals) ~= 56
        error( ...
            "Domain %s: expected 56 values after reordering, received %d.", ...
            domains(d), ...
            numel(vals));
    end

    if any(~isfinite(vals),"all")
        bad = baseFeatureNames(~isfinite(vals));
        error( ...
            "Domain %s: non-finite feature(s): %s", ...
            domains(d), ...
            strjoin(bad,", "));
    end

    if logical(R.Resample)
        error( ...
            "Domain %s: unexpected spatial resampling.", ...
            domains(d));
    end

    idx = (d-1)*56 + (1:56);

    features(idx) = vals;
    featureNames(idx) = domains(d) + "_" + baseFeatureNames;

    details.Rows(d) = size(data,1);
    details.Cols(d) = size(data,2);
    details.Min(d) = double(min(data(:)));
    details.Max(d) = double(max(data(:)));
    details.Resample(d) = logical(R.Resample);
    details.ReturnedOrderMatchedRequested(d) = ...
        isequal(returnedNames(:),baseFeatureNames(:));

end

assert(numel(features) == 280, ...
    "The extractor must return exactly 280 descriptors.");

assert(all(isfinite(features)), ...
    "All extracted descriptors must be finite.");

end