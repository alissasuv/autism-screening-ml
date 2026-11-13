#pragma once

#include "preprocessing.hpp"

#include <cstddef>
#include <optional>
#include <span>

enum class SplitCriterion {
    Variance,
    Gini
};

struct Split {
    std::size_t feature = 0;
    double threshold = 0.0;
    double score = 0.0;
};

struct FeatureSplitResult {
    bool is_constant = true;
    std::optional<Split> best;
};

class SplitFinder {
public:
    std::optional<Split> find_best(
        const NumericMatrix& features,
        std::span<const double> target,
        std::span<const std::size_t> object_ids,
        std::span<const std::size_t> feature_ids,
        SplitCriterion criterion,
        std::size_t min_samples_leaf = 1,
        std::span<const std::size_t> counts = {}
    ) const;

    FeatureSplitResult find_for_feature(
        const NumericMatrix& features,
        std::span<const double> target,
        std::span<const std::size_t> object_ids,
        std::size_t feature,
        SplitCriterion criterion,
        std::size_t min_samples_leaf = 1,
        std::span<const std::size_t> counts = {}
    ) const;
};
