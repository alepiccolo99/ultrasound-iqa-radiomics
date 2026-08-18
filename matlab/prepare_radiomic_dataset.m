function summary = prepare_radiomic_dataset(manifestFile, roiFile, referenceDir, testDir, outputDir)
%PREPARE_RADIOMIC_DATASET Extract reference/test radiomic features for a custom dataset.
%
% This dataset-level wrapper applies the same radiomic extraction used in
% the paper to an arbitrary number of reference groups and associated test
% images. One rectangular ROI is defined per reference group and the same
% ROI coordinates are applied to that reference image and all test images
% paired with it.
%
% REQUIRED INPUT FILES
%
% manifestFile CSV columns:
%   SampleID, ReferenceID, ReferenceFile, TestFile
%
% roiFile CSV columns:
%   ReferenceID, X, Y, Width, Height
%
% INPUT DIRECTORIES
%   referenceDir : directory containing the reference images named in
%                  ReferenceFile
%   testDir      : directory containing the test images named in TestFile
%
% OUTPUTS
%   reference_radiomic_features.csv
%   test_radiomic_features.csv
%   radiomic_dataset.mat  (exact doubles + scipy-compatible metadata)
%
% The CSV feature tables can be combined with a subjective_scores.csv file
% and passed to the Python grouped nested-validation script in CSV mode.
%
% Radiomic extraction is delegated to extract_radiomic_features.m:
%   - 56 base features
%   - native B-mode + Haar L1 LL/LH/HL/HH
%   - FixedBinNumber = 32
%   - no resegmentation
%   - no spatial resampling
%   - signed wavelet coefficients retained
%
% MATLAB R2026a with Medical Imaging Toolbox and Wavelet Toolbox is
% recommended for reproduction of the paper extraction environment.

arguments
    manifestFile (1,1) string
    roiFile (1,1) string
    referenceDir (1,1) string
    testDir (1,1) string
    outputDir (1,1) string
end

%% Validate paths
assert(isfile(manifestFile), "Manifest file not found: %s", manifestFile);
assert(isfile(roiFile), "ROI file not found: %s", roiFile);
assert(isfolder(referenceDir), "Reference-image directory not found: %s", referenceDir);
assert(isfolder(testDir), "Test-image directory not found: %s", testDir);

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% Read manifest
opts = detectImportOptions(manifestFile, "VariableNamingRule", "preserve");
requiredManifest = ["SampleID","ReferenceID","ReferenceFile","TestFile"];

missing = setdiff(requiredManifest, string(opts.VariableNames), "stable");
if ~isempty(missing)
    error("Manifest is missing required column(s): %s", strjoin(missing, ", "));
end

stringVars = intersect(["SampleID","ReferenceID","ReferenceFile","TestFile"], ...
    string(opts.VariableNames), "stable");
opts = setvartype(opts, stringVars, "string");
manifest = readtable(manifestFile, opts);

assert(height(manifest) >= 1, "Manifest must contain at least one test sample.");
assert(numel(unique(manifest.SampleID)) == height(manifest), ...
    "SampleID values must be unique.");

if any(ismissing(manifest.ReferenceID)) || ...
   any(ismissing(manifest.ReferenceFile)) || ...
   any(ismissing(manifest.TestFile))
    error("Manifest contains missing ReferenceID, ReferenceFile, or TestFile values.");
end

%% Read ROI table
roiOpts = detectImportOptions(roiFile, "VariableNamingRule", "preserve");
requiredROI = ["ReferenceID","X","Y","Width","Height"];

missing = setdiff(requiredROI, string(roiOpts.VariableNames), "stable");
if ~isempty(missing)
    error("ROI table is missing required column(s): %s", strjoin(missing, ", "));
end

roiOpts = setvartype(roiOpts, "ReferenceID", "string");
roiTable = readtable(roiFile, roiOpts);

if any(ismissing(roiTable.ReferenceID))
    error("ROI table contains missing ReferenceID values.");
end

roiValues = roiTable{:, ["X","Y","Width","Height"]};
if ~isnumeric(roiValues) || any(~isfinite(roiValues), "all")
    error("ROI coordinates must be finite numeric values.");
end
if any(roiTable.Width < 0) || any(roiTable.Height < 0)
    error("ROI Width and Height must be non-negative.");
end

%% Determine reference groups
referenceLabels = unique(manifest.ReferenceID, "stable");
nGroups = numel(referenceLabels);
nSamples = height(manifest);

[matchedGroups, sampleGroupIndex] = ismember(manifest.ReferenceID, referenceLabels);
if ~all(matchedGroups)
    error("Could not map every sample to a reference group.");
end
sampleGroupIndex = uint32(sampleGroupIndex);

if nGroups < 1
    error("No reference groups were found in the manifest.");
end

% Every manifest ReferenceID must have exactly one ROI row.
for g = 1:nGroups
    refID = referenceLabels(g);

    roiRows = find(roiTable.ReferenceID == refID);
    if numel(roiRows) ~= 1
        error("ReferenceID '%s' must have exactly one ROI row; found %d.", ...
            refID, numel(roiRows));
    end

    refFiles = unique(manifest.ReferenceFile(manifest.ReferenceID == refID));
    if numel(refFiles) ~= 1
        error("ReferenceID '%s' must map to exactly one ReferenceFile; found %d.", ...
            refID, numel(refFiles));
    end
end

extraROI = setdiff(roiTable.ReferenceID, referenceLabels, "stable");
if ~isempty(extraROI)
    warning("ROI table contains unused ReferenceID value(s): %s", ...
        strjoin(extraROI, ", "));
end

%% Frozen feature identity
baseNames = radiomic_feature_names();
domains = ["O","LL","LH","HL","HH"];

allFeatureNames = strings(1, numel(baseNames) * numel(domains));
q = 0;
for d = 1:numel(domains)
    for f = 1:numel(baseNames)
        q = q + 1;
        allFeatureNames(q) = domains(d) + "_" + baseNames(f);
    end
end

assert(numel(allFeatureNames) == 280, ...
    "Expected exactly 280 output radiomic descriptors.");

%% Extract each reference once
X_reference = nan(nGroups, 280);
referenceFiles = strings(nGroups, 1);
referenceCropRows = nan(nGroups, 1);
referenceCropCols = nan(nGroups, 1);

fprintf("Extracting %d reference image(s)...\n", nGroups);

for g = 1:nGroups
    refID = referenceLabels(g);

    groupRows = find(manifest.ReferenceID == refID);
    refFile = unique(manifest.ReferenceFile(groupRows));
    refFile = refFile(1);
    referenceFiles(g) = refFile;

    refPath = fullfile(referenceDir, refFile);
    assert(isfile(refPath), "Reference image not found: %s", refPath);

    roiRow = find(roiTable.ReferenceID == refID, 1);
    roi = double(roiTable{roiRow, ["X","Y","Width","Height"]});

    I = read_grayscale_image(refPath);
    imgROI = imcrop(I, roi);

    if isempty(imgROI)
        error("ReferenceID '%s': ROI produced an empty crop.", refID);
    end

    [x, names] = extract_radiomic_features(imgROI);

    if ~isequal(names(:), allFeatureNames(:))
        error("ReferenceID '%s': radiomic feature-name/order mismatch.", refID);
    end

    X_reference(g, :) = x;
    referenceCropRows(g) = size(imgROI, 1);
    referenceCropCols(g) = size(imgROI, 2);

    fprintf("  [%d/%d] %s | crop=%dx%d\n", ...
        g, nGroups, refID, size(imgROI,1), size(imgROI,2));
end

if any(~isfinite(X_reference), "all")
    error("Reference extraction produced non-finite radiomic values.");
end

%% Extract tests
X_test = nan(nSamples, 280);
testCropRows = nan(nSamples, 1);
testCropCols = nan(nSamples, 1);

fprintf("Extracting %d test image(s)...\n", nSamples);

for i = 1:nSamples
    refID = manifest.ReferenceID(i);
    testFile = manifest.TestFile(i);

    groupIdx = find(referenceLabels == refID, 1);
    if isempty(groupIdx)
        error("SampleID '%s': unknown ReferenceID '%s'.", ...
            string(manifest.SampleID(i)), refID);
    end

    testPath = fullfile(testDir, testFile);
    assert(isfile(testPath), "Test image not found: %s", testPath);

    roiRow = find(roiTable.ReferenceID == refID, 1);
    roi = double(roiTable{roiRow, ["X","Y","Width","Height"]});

    I = read_grayscale_image(testPath);
    imgROI = imcrop(I, roi);

    if isempty(imgROI)
        error("SampleID '%s': ROI produced an empty crop.", ...
            string(manifest.SampleID(i)));
    end

    if size(imgROI,1) ~= referenceCropRows(groupIdx) || ...
       size(imgROI,2) ~= referenceCropCols(groupIdx)
        error( ...
            "SampleID '%s': test crop %dx%d does not match paired reference crop %dx%d.", ...
            string(manifest.SampleID(i)), ...
            size(imgROI,1), size(imgROI,2), ...
            referenceCropRows(groupIdx), referenceCropCols(groupIdx));
    end

    [x, names] = extract_radiomic_features(imgROI);

    if ~isequal(names(:), allFeatureNames(:))
        error("SampleID '%s': radiomic feature-name/order mismatch.", ...
            string(manifest.SampleID(i)));
    end

    X_test(i, :) = x;
    testCropRows(i) = size(imgROI, 1);
    testCropCols(i) = size(imgROI, 2);

    if i == 1 || mod(i,10) == 0 || i == nSamples
        fprintf("  [%d/%d] SampleID=%s\n", ...
            i, nSamples, string(manifest.SampleID(i)));
    end
end

if any(~isfinite(X_test), "all")
    error("Test extraction produced non-finite radiomic values.");
end

%% Export feature tables
referenceMeta = table(referenceLabels, referenceFiles, ...
    'VariableNames', {'ReferenceID','ReferenceFile'});
referenceFeatureTable = array2table( ...
    X_reference, 'VariableNames', cellstr(allFeatureNames));
referenceOutput = [referenceMeta referenceFeatureTable];

testMeta = manifest(:, ["SampleID","ReferenceID"]);
testFeatureTable = array2table( ...
    X_test, 'VariableNames', cellstr(allFeatureNames));
testOutput = [testMeta testFeatureTable];

referenceCsv = fullfile(outputDir, "reference_radiomic_features.csv");
testCsv = fullfile(outputDir, "test_radiomic_features.csv");
matFile = fullfile(outputDir, "radiomic_dataset.mat");

writetable(referenceOutput, referenceCsv);
writetable(testOutput, testCsv);

% Store exact MATLAB double values plus plain MATLAB primitives that can be
% read directly by scipy.io.loadmat. Avoid MATLAB string/table objects here,
% because they are serialized as MCOS objects in MAT v7 files.
sample_id = cellstr(string(manifest.SampleID));
reference_labels = cellstr(referenceLabels);
reference_files = cellstr(referenceFiles);
feature_names = cellstr(allFeatureNames(:));
group_index = sampleGroupIndex;

reference_crop_rows = referenceCropRows;
reference_crop_cols = referenceCropCols;
test_crop_rows = testCropRows;
test_crop_cols = testCropCols;

assert(numel(group_index) == nSamples, ...
    "Internal error: group_index length does not match the number of samples.");

save(matFile, ...
    "X_reference", "X_test", ...
    "sample_id", "group_index", ...
    "reference_labels", "reference_files", ...
    "feature_names", ...
    "reference_crop_rows", "reference_crop_cols", ...
    "test_crop_rows", "test_crop_cols", ...
    "-v7");

%% Final validation
assert(height(referenceOutput) == nGroups);
assert(height(testOutput) == nSamples);
assert(width(referenceOutput) == 282); % ReferenceID + ReferenceFile + 280 features
assert(width(testOutput) == 282);      % SampleID + ReferenceID + 280 features

summary = struct();
summary.ReferenceGroups = nGroups;
summary.TestSamples = nSamples;
summary.FeaturesPerImage = 280;
summary.ReferenceFeatureCSV = string(referenceCsv);
summary.TestFeatureCSV = string(testCsv);
summary.MATFile = string(matFile);

fprintf("\nRADIOMIC DATASET EXTRACTION: PASS\n");
fprintf("Reference groups:  %d\n", nGroups);
fprintf("Test samples:      %d\n", nSamples);
fprintf("Features/image:    280\n");
fprintf("Reference CSV:     %s\n", referenceCsv);
fprintf("Test CSV:          %s\n", testCsv);
fprintf("Exact MAT data:    %s\n", matFile);
end


function I = read_grayscale_image(path)
%READ_GRAYSCALE_IMAGE Read an image and convert RGB/RGBA input to grayscale.

I = imread(path);

if ndims(I) == 3
    if size(I,3) == 3
        I = im2gray(I);
    elseif size(I,3) == 4
        I = im2gray(I(:,:,1:3));
    else
        error("Unsupported multi-channel image with %d channels: %s", ...
            size(I,3), path);
    end
end

if ndims(I) ~= 2
    error("Image must resolve to a 2-D grayscale matrix: %s", path);
end

if ~isnumeric(I) && ~islogical(I)
    error("Unsupported image data type in: %s", path);
end
end
