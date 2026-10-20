#include "dataset.hpp"

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

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