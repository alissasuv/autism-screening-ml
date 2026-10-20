#include "preprocessing.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

constexpr const char* missing_category = "__MISSING__";

double get_fill_value(
    NumericImputation strategy,
    const NumericStatistics& stats
) {
    switch (strategy) {
        case NumericImputation::Mean:
            return stats.mean;

        case NumericImputation::Median:
            return stats.q50;

        case NumericImputation::Zero:
            return 0.0;
    }

    throw std::runtime_error(
        "Unknown imputation strategy"
    );
}

double scale_value(
    double value,
    NumericScaling strategy,
    const NumericStatistics& stats
) {
    double numerator = 0.0;
    double denominator = 0.0;

    switch (strategy) {
        case NumericScaling::MinMax:
            numerator = value - stats.min;
            denominator = stats.max - stats.min;
            break;

        case NumericScaling::Standard:
            numerator = value - stats.mean;
            denominator = std::sqrt(stats.variance);
            break;

        case NumericScaling::Robust:
            numerator = value - stats.q50;
            denominator = stats.q75 - stats.q25;
            break;

        default:
            throw std::runtime_error(
                "Unknown scaling strategy"
            );
    }

    if (denominator == 0.0) {
        return 0.0;
    }

    return numerator / denominator;
}

std::vector<double> process_numeric_column(
    const NumericColumn& column,
    const NumericStatistics& stats,
    const PreprocessingOptions& options
) {
    const double fill_value =
        get_fill_value(options.imputation, stats);

    std::vector<double> processed;
    processed.reserve(column.values.size());

    for (const auto& cell : column.values) {
        const double value =
            cell.has_value() ? *cell : fill_value;

        processed.push_back(
            scale_value(value, options.scaling, stats)
        );
    }

    return processed;
}

std::vector<std::string> get_categories(
    const CategoricalColumn& column
) {
    std::set<std::string> unique_categories;
    bool has_missing = false;

    for (const auto& cell : column.values) {
        if (cell.has_value()) {
            unique_categories.insert(*cell);
        } else {
            has_missing = true;
        }
    }

    if (has_missing &&
        unique_categories.contains(missing_category)) {
        throw std::runtime_error(
            "Missing category marker conflicts with existing data"
        );
    }

    std::vector<std::string> categories(
        unique_categories.begin(),
        unique_categories.end()
    );

    if (has_missing) {
        categories.push_back(missing_category);
    }

    return categories;
}

std::string get_category_value(
    const std::optional<std::string>& cell
) {
    return cell.has_value()
        ? *cell
        : missing_category;
}

} // namespace

NumericMatrix preprocess_dataset(
    const Dataset& dataset,
    const std::vector<std::optional<NumericStatistics>>& statistics,
    const PreprocessingOptions& options
) {
    if (statistics.size() != dataset.columns.size() ||
        dataset.column_names.size() != dataset.columns.size()) {
        throw std::runtime_error(
            "Dataset metadata has inconsistent column counts"
        );
    }

    NumericMatrix matrix;

    std::vector<std::vector<double>> output_columns;
    std::vector<std::string> output_names;

    for (std::size_t i = 0; i < dataset.columns.size(); ++i) {
        const auto& column = dataset.columns[i];
        const auto& name = dataset.column_names[i];

        const auto column_size = std::visit(
            [](const auto& typed_column) {
                return typed_column.values.size();
            },
            column
        );

        if (column_size != dataset.row_count) {
            throw std::runtime_error(
                "Inconsistent column length: " + name
            );
        }

        // Keep the target separate from predictor features.
        if (name == "Class/ASD") {
            const auto* target =
                std::get_if<CategoricalColumn>(&column);

            if (target == nullptr) {
                throw std::runtime_error(
                    "Target column must be categorical"
                );
            }

            matrix.target_values.reserve(target->values.size());

            for (const auto& cell : target->values) {
                if (!cell.has_value()) {
                    throw std::runtime_error(
                        "Target column contains missing values"
                    );
                }

                matrix.target_values.push_back(*cell);
            }

            continue;
        }

        if (const auto* numeric =
                std::get_if<NumericColumn>(&column)) {

            if (!statistics[i].has_value()) {
                throw std::runtime_error(
                    "Missing numerical statistics for: " + name
                );
            }

            output_columns.push_back(
                process_numeric_column(
                    *numeric,
                    *statistics[i],
                    options
                )
            );

            output_names.push_back(name);
            continue;
        }

        const auto& categorical =
            std::get<CategoricalColumn>(column);

        const auto categories = get_categories(categorical);

        if (options.encoding == CategoryEncoding::Ordinal) {
            std::vector<double> encoded;
            encoded.reserve(categorical.values.size());

            for (const auto& cell : categorical.values) {
                const std::string value =
                    get_category_value(cell);

                const auto it = std::find(
                    categories.begin(),
                    categories.end(),
                    value
                );

                if (it == categories.end()) {
                    throw std::runtime_error(
                        "Unknown category: " + value
                    );
                }

                encoded.push_back(
                    static_cast<double>(
                        std::distance(categories.begin(), it)
                    )
                );
            }

            output_columns.push_back(std::move(encoded));
            output_names.push_back(name);

        } else if (
            options.encoding == CategoryEncoding::OneHot
        ) {
            for (const auto& category : categories) {
                std::vector<double> encoded;
                encoded.reserve(categorical.values.size());

                for (const auto& cell : categorical.values) {
                    encoded.push_back(
                        get_category_value(cell) == category
                            ? 1.0
                            : 0.0
                    );
                }

                output_columns.push_back(std::move(encoded));
                output_names.push_back(
                    name + "_" + category
                );
            }

        } else {
            throw std::runtime_error(
                "Unknown category encoding strategy"
            );
        }
    }

    if (matrix.target_values.size() != dataset.row_count) {
        throw std::runtime_error(
            "Target column Class/ASD is missing"
        );
    }

    matrix.rows = dataset.row_count;
    matrix.cols = output_columns.size();
    matrix.column_names = std::move(output_names);

    matrix.values.reserve(matrix.rows * matrix.cols);

    // Convert column-oriented data to row-major matrix.
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            matrix.values.push_back(output_columns[col][row]);
        }
    }

    return matrix;
}