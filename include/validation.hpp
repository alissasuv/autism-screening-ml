#pragma once

#include "csv_reader.hpp"
#include "metrics.hpp"
#include "model_factory.hpp"
#include "preprocessing.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lab5 {

struct Fold {
    std::vector<std::size_t> train_indices;
    std::vector<std::size_t> validation_indices;
};

struct CrossValidationResult {
    std::vector<double> fold_scores;
    double mean = 0.0;
    double stddev = 0.0;
};

// Use the same folds for all model candidates.
std::vector<Fold> make_folds(
    const RawDataset& data,
    const std::string& target_name,
    Task task,
    std::size_t n_splits,
    std::uint32_t random_state
);

// Preserve the raw values and column schema.
RawDataset select_raw_rows(
    const RawDataset& data,
    const std::vector<std::size_t>& indices
);

// Only the training dataset is passed to cross-validation.
CrossValidationResult cross_validate(
    const RawDataset& train,
    const std::string& target_name,
    const PreprocessingOptions& preprocessing,
    const ModelConfig& model,
    Scoring scoring,
    const std::vector<Fold>& folds
);

} // namespace lab5
