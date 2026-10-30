#include "dataset.hpp"

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <variant>
#include <type_traits>

namespace {

double parse_number(const std::string& text) {
    double number = 0.0;

    const char* begin = text.data();
    const char* end = begin + text.size();

    const auto [ptr, error] =
        std::from_chars(begin, end, number);

    if (error != std::errc{} ||
        ptr != end ||
        !std::isfinite(number)) {
        throw std::runtime_error(
            "Invalid numeric value: " + text
        );
    }

    return number;
}

} // namespace

Dataset make_typed_dataset(
    const RawDataset& raw,
    const std::vector<FeatureType>& types
) {
    if (raw.columns.size() != types.size() ||
        raw.columns.size() != raw.column_names.size()) {
        throw std::runtime_error(
            "Inconsistent dataset metadata"
        );
    }

    Dataset dataset;
    dataset.column_names = raw.column_names;
    dataset.row_count = raw.row_count;
    dataset.columns.reserve(raw.columns.size());

    for (std::size_t i = 0; i < raw.columns.size(); ++i) {
        const auto& source = raw.columns[i];

        if (source.size() != raw.row_count) {
            throw std::runtime_error(
                "Inconsistent column length: " +
                raw.column_names[i]
            );
        }

        if (types[i] == FeatureType::Numerical) {
            NumericColumn column;
            column.values.reserve(source.size());

            for (const auto& cell : source) {
                if (cell.has_value()) {
                    column.values.emplace_back(
                        parse_number(*cell)
                    );
                } else {
                    column.values.emplace_back(std::nullopt);
                }
            }

            dataset.columns.emplace_back(std::move(column));
        } else {
            CategoricalColumn column;
            column.values = source;

            dataset.columns.emplace_back(std::move(column));
        }
    }

    return dataset;
}

std::vector<double> extract_numeric_values(
    const NumericColumn& column
) {
    std::vector<double> values;
    values.reserve(column.values.size());

    for (const auto& cell : column.values) {
        if (cell.has_value()) {
            values.push_back(*cell);
        }
    }

    return values;
}

DatasetSplit split_dataset(
    const Dataset& dataset,
    double train_ratio,
    unsigned int seed
) {
    if (!std::isfinite(train_ratio) ||
        train_ratio <= 0.0 ||
        train_ratio >= 1.0) {
        throw std::invalid_argument(
            "Train ratio must be between 0 and 1"
        );
    }

    if (dataset.row_count < 2) {
        throw std::invalid_argument(
            "Dataset must contain at least two rows"
        );
    }

    if (dataset.columns.size() != dataset.column_names.size()) {
        throw std::invalid_argument(
            "Inconsistent dataset metadata"
        );
    }

    for (const auto& column : dataset.columns) {
        const std::size_t size = std::visit(
            [](const auto& typed_column) {
                return typed_column.values.size();
            },
            column
        );

        if (size != dataset.row_count) {
            throw std::invalid_argument(
                "Inconsistent dataset column length"
            );
        }
    }

    const std::size_t train_size =
        static_cast<std::size_t>(dataset.row_count * train_ratio);

    if (train_size == 0 || train_size == dataset.row_count) {
        throw std::invalid_argument(
            "Train ratio produces an empty subset"
        );
    }

    std::vector<std::size_t> indices(dataset.row_count);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 generator(seed);
    std::shuffle(indices.begin(), indices.end(), generator);

    DatasetSplit result;

    result.train.column_names = dataset.column_names;
    result.test.column_names = dataset.column_names;

    result.train.row_count = train_size;
    result.test.row_count = dataset.row_count - train_size;

    result.train.columns.reserve(dataset.columns.size());
    result.test.columns.reserve(dataset.columns.size());

    for (const auto& column : dataset.columns) {
        std::visit(
            [&](const auto& source) {
                using ColumnType = std::decay_t<decltype(source)>;

                ColumnType train_column;
                ColumnType test_column;

                train_column.values.reserve(result.train.row_count);
                test_column.values.reserve(result.test.row_count);

                for (std::size_t i = 0; i < indices.size(); ++i) {
                    const auto& value = source.values[indices[i]];

                    if (i < train_size) {
                        train_column.values.push_back(value);
                    } else {
                        test_column.values.push_back(value);
                    }
                }

                result.train.columns.emplace_back(
                    std::move(train_column)
                );

                result.test.columns.emplace_back(
                    std::move(test_column)
                );
            },
            column
        );
    }

    return result;
}