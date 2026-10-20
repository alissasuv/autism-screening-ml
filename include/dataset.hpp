#pragma once

#include "csv_reader.hpp"
#include "feature_type.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <variant>
#include <vector>

struct NumericColumn {
    std::vector<std::optional<double>> values;
};

struct CategoricalColumn {
    std::vector<std::optional<std::string>> values;
};

using Column = std::variant<NumericColumn, CategoricalColumn>;

struct Dataset {
    std::vector<std::string> column_names;
    std::vector<Column> columns;
    std::size_t row_count = 0;
};

Dataset make_typed_dataset(
    const RawDataset& raw,
    const std::vector<FeatureType>& types
);