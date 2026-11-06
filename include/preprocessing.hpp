#pragma once

#include "dataset.hpp"
#include "statistics.hpp"
#include "matrix_slice.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>
#include <span>
#include <stdexcept>

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

struct PreprocessingState {
    PreprocessingOptions options;

    std::vector<std::optional<NumericStatistics>> numeric_statistics;

    std::vector<std::vector<std::string>> categories;

    std::vector<std::string> column_names;
};

PreprocessingState fit_preprocessing(
    const Dataset& train,
    const PreprocessingOptions& options
);

enum class MatrixLayout {
    RowMajor,
    ColumnMajor
};

struct NumericMatrix {
    std::size_t rows = 0;
    std::size_t cols = 0;

    MatrixLayout layout = MatrixLayout::RowMajor;

    std::vector<std::string> column_names;
    std::vector<double> values;
    std::vector<std::string> target_values;

    // Legacy row access, retained for compatibility with Lab 2.
    std::span<double> operator[](std::size_t row) {
        if (row >= rows) {
            throw std::out_of_range("Matrix row index out of range");
        }

        if (layout == MatrixLayout::ColumnMajor && cols > 1) {
            throw std::logic_error(
                "Cannot create a contiguous row span in ColumnMajor layout"
            );
        }

        if (cols == 0) {
            return {};
        }

        if (layout == MatrixLayout::RowMajor) {
            return {values.data() + row * cols, cols};
        }

        return {values.data() + row, cols};
    }

    std::span<const double> operator[](std::size_t row) const {
        if (row >= rows) {
            throw std::out_of_range("Matrix row index out of range");
        }

        if (layout == MatrixLayout::ColumnMajor && cols > 1) {
            throw std::logic_error(
                "Cannot create a contiguous row span in ColumnMajor layout"
            );
        }

        if (cols == 0) {
            return {};
        }

        if (layout == MatrixLayout::RowMajor) {
            return {values.data() + row * cols, cols};
        }

        return {values.data() + row, cols};
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

    // Element access.
    double& operator()(std::size_t row, std::size_t col) &;
    const double& operator()(std::size_t row, std::size_t col) const &;

    // Row slices.
    MatrixSlice<double>
    operator()(std::size_t row, std::nullopt_t) &;

    MatrixSlice<const double>
    operator()(std::size_t row, std::nullopt_t) const &;

    // Column slices.
    MatrixSlice<double>
    operator()(std::nullopt_t, std::size_t col) &;

    MatrixSlice<const double>
    operator()(std::nullopt_t, std::size_t col) const &;

    // Disallow element access from temporary matrices.
    double& operator()(std::size_t, std::size_t) && = delete;
    const double& operator()(std::size_t, std::size_t) const && = delete;

    // Disallow row slices from temporary matrices.
    MatrixSlice<double>
    operator()(std::size_t, std::nullopt_t) && = delete;

    MatrixSlice<const double>
    operator()(std::size_t, std::nullopt_t) const && = delete;

    // Disallow column slices from temporary matrices.
    MatrixSlice<double>
    operator()(std::nullopt_t, std::size_t) && = delete;

    MatrixSlice<const double>
    operator()(std::nullopt_t, std::size_t) const && = delete;
};

NumericMatrix preprocess_dataset(
    const Dataset& dataset,
    const std::vector<std::optional<NumericStatistics>>& statistics,
    const PreprocessingOptions& options
);

std::vector<double> encode_target(
    const std::vector<std::string>& target_values
);

NumericMatrix transform_preprocessing(
    const Dataset& dataset,
    const PreprocessingState& state
);
