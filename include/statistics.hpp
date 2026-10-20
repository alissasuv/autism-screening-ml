#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

struct NumericStatistics {
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double variance = 0.0;
    double q05 = 0.0;
    double q25 = 0.0;
    double q50 = 0.0;
    double q75 = 0.0;
    double q95 = 0.0;
};

struct CategoricalStatistics {
    std::map<std::string, std::size_t> frequencies;
};

std::vector<double> extract_numeric_values(
    const std::vector<std::optional<std::string>>& column
);

NumericStatistics calculate_numeric_statistics(
    std::span<const double> values
);

CategoricalStatistics calculate_categorical_statistics(
    const std::vector<std::optional<std::string>>& column
);

void print_numeric_histogram(
    std::span<const double> values,
    std::size_t bin_count
);

void print_categorical_histogram(
    const CategoricalStatistics& statistics
);
