#include "statistics.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

double parse_number(const std::string& text) {
    double value = 0.0;

    const char* begin = text.data();
    const char* end = begin + text.size();

    auto [ptr, error] = std::from_chars(begin, end, value);

    if (error != std::errc{} || ptr != end ||
        !std::isfinite(value)) {
        throw std::runtime_error(
            "Invalid numeric value: " + text
        );
    }

    return value;
}

double quantile(std::span<const double> sorted, double p) {
    const double position = p * (sorted.size() - 1);

    const auto lower =
        static_cast<std::size_t>(std::floor(position));

    const auto upper =
        static_cast<std::size_t>(std::ceil(position));

    const double fraction = position - lower;

    return sorted[lower] +
           fraction * (sorted[upper] - sorted[lower]);
}

void print_bar(std::size_t count, std::size_t maximum) {
    constexpr std::size_t max_width = 40;

    const std::size_t width =
        maximum == 0 ? 0 : count * max_width / maximum;

    std::cout << std::string(width, '#')
              << " " << count << '\n';
}

} // namespace

std::vector<double> extract_numeric_values(
    const std::vector<std::optional<std::string>>& column
) {
    std::vector<double> values;
    values.reserve(column.size());

    for (const auto& cell : column) {
        if (cell.has_value()) {
            values.push_back(parse_number(*cell));
        }
    }

    return values;
}

NumericStatistics calculate_numeric_statistics(
    std::span<const double> values
) {
    if (values.empty()) {
        throw std::runtime_error(
            "Cannot calculate statistics for an empty column"
        );
    }

    std::vector<double> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());

    NumericStatistics stats;

    stats.min = sorted.front();
    stats.max = sorted.back();

    stats.mean = std::accumulate(
        values.begin(), values.end(), 0.0
    ) / values.size();

    double squared_sum = 0.0;

    for (double value : values) {
        const double difference = value - stats.mean;
        squared_sum += difference * difference;
    }

    stats.variance = squared_sum / values.size();

    stats.q05 = quantile(sorted, 0.05);
    stats.q25 = quantile(sorted, 0.25);
    stats.q50 = quantile(sorted, 0.50);
    stats.q75 = quantile(sorted, 0.75);
    stats.q95 = quantile(sorted, 0.95);

    return stats;
}

CategoricalStatistics calculate_categorical_statistics(
    const std::vector<std::optional<std::string>>& column
) {
    CategoricalStatistics stats;

    for (const auto& cell : column) {
        if (cell.has_value()) {
            ++stats.frequencies[*cell];
        }
    }

    return stats;
}

void print_numeric_histogram(
    std::span<const double> values,
    std::size_t bin_count
) {
    if (values.empty() || bin_count == 0) {
        return;
    }

    const auto [min_it, max_it] =
        std::minmax_element(values.begin(), values.end());

    const double min = *min_it;
    const double max = *max_it;

    std::vector<std::size_t> bins(bin_count, 0);

    if (min == max) {
        bins[0] = values.size();
    } else {
        const double step = (max - min) / bin_count;

        for (double value : values) {
            std::size_t index =
                static_cast<std::size_t>((value - min) / step);

            if (index >= bin_count) {
                index = bin_count - 1;
            }

            ++bins[index];
        }
    }

    const std::size_t maximum =
        *std::max_element(bins.begin(), bins.end());

    std::cout << "\nHistogram:\n";

    for (std::size_t i = 0; i < bin_count; ++i) {
        double left = min;
        double right = max;

        if (min != max) {
            const double step = (max - min) / bin_count;
            left = min + i * step;
            right = (i + 1 == bin_count)
                ? max
                : min + (i + 1) * step;
        }

        std::cout << "[" << std::fixed
                  << std::setprecision(2)
                  << left << ", " << right
                  << (i + 1 == bin_count ? "] " : ") ");

        print_bar(bins[i], maximum);
    }
}

void print_categorical_histogram(
    const CategoricalStatistics& statistics
) {
    std::size_t maximum = 0;

    for (const auto& [category, count] :
         statistics.frequencies) {
        maximum = std::max(maximum, count);
    }

    std::cout << "\nCategories:\n";

    for (const auto& [category, count] :
         statistics.frequencies) {
        std::cout << category << ": ";
        print_bar(count, maximum);
    }
}
