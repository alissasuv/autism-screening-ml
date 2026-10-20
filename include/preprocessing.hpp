#pragma once

#include "dataset.hpp"
#include "statistics.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

enum class NumericImputation {
    Mean,
    Median,
    Zero
};

enum class NumericScaling {
    MinMax,
    Standard,
    Robust
};

enum class CategoryEncoding {
    OneHot,
    Ordinal
};

struct PreprocessingOptions {
    NumericImputation imputation = NumericImputation::Median;
    NumericScaling scaling = NumericScaling::MinMax;
    CategoryEncoding encoding = CategoryEncoding::OneHot;
};

struct NumericMatrix {
    std::size_t rows = 0;
    std::size_t cols = 0;

    std::vector<std::string> column_names;
    std::vector<double> values;
    std::vector<std::string> target_values;
};

NumericMatrix preprocess_dataset(
    const Dataset& dataset,
    const std::vector<std::optional<NumericStatistics>>& statistics,
    const PreprocessingOptions& options
);