#pragma once

#include "evaluation.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lab5 {

struct FeatureImportance {
    std::size_t feature_index = 0;
    std::string feature_name;
    double mean_importance = 0.0;
    double stddev = 0.0;
    std::vector<double> repeat_importances;
};

struct PermutationImportanceResult {
    double baseline_score = 0.0;
    std::vector<FeatureImportance> features;
};

PermutationImportanceResult permutation_feature_importance(
    const TrainedEvaluation& trained,
    const RawDataset& test,
    std::size_t repeats,
    std::uint32_t random_state
);

void save_feature_importance(
    const PermutationImportanceResult& result,
    const std::string& filename
);

} // namespace lab5





