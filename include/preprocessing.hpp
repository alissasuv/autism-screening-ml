#pragma once

#include "dataset.hpp"
#include "statistics.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>
#include <span>
#include <stdexcept>
#include <string>

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

    std::span<double> operator[](std::size_t row) {
        return {values.data() + row * cols, cols};
    }

    std::span<const double> operator[](std::size_t row) const {
        return {values.data() + row * cols, cols};
    }

    std::span<double> at(std::size_t row) {
        if (row >= rows) {
            throw std::out_of_range("Matrix row index out of range");
        }
        return (*this)[row];
    }

    std::span<const double> at(std::size_t row) const {
        if (row >= rows) {
            throw std::out_of_range("Matrix row index out of range");
        }
        return (*this)[row];
    }
};

NumericMatrix preprocess_dataset(
    const Dataset& dataset,
    const std::vector<std::optional<NumericStatistics>>& statistics,
    const PreprocessingOptions& options
);