#pragma once

#include "csv_reader.hpp"
#include "feature_type.hpp"
#include "preprocessing.hpp"

#include <string>
#include <vector>

namespace lab5 {

struct FittedRawPreprocessing {
    std::vector<std::string> column_names;
    std::vector<FeatureType> column_types;
    std::string target_name;
    PreprocessingState state;
};

// Learn feature types, statistics and category dictionaries
// exclusively from the training data.
FittedRawPreprocessing fit_raw_preprocessing(
    const RawDataset& train,
    const std::string& target_name,
    const PreprocessingOptions& options
);

// Transform data using the previously fitted state.
// Never infer feature types or recompute statistics here.
NumericMatrix transform_raw_preprocessing(
    const RawDataset& data,
    const FittedRawPreprocessing& fitted
);

// Load preprocessing settings from a YAML configuration.
PreprocessingOptions load_preprocessing_config(
    const std::string& filename
);

} // namespace lab5


