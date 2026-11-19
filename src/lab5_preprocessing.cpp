#include "lab5_preprocessing.hpp"

#include "dataset.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lab5 {

namespace {

constexpr const char* legacy_target_name = "Class/ASD";

void validate_raw(
    const RawDataset& data,
    bool allow_empty
) {
    if ((!allow_empty && data.row_count == 0) ||
        data.column_names.empty() ||
        data.column_names.size() != data.columns.size()) {
        throw std::invalid_argument(
            "Invalid raw dataset dimensions"
        );
    }

    std::set<std::string> names;

    for (std::size_t col = 0;
         col < data.columns.size();
         ++col) {

        const auto& name = data.column_names[col];

        if (name.empty() || !names.insert(name).second) {
            throw std::invalid_argument(
                "Empty or duplicate column name: " + name
            );
        }

        if (data.columns[col].size() != data.row_count) {
            throw std::invalid_argument(
                "Inconsistent column length: " + name
            );
        }
    }
}

std::size_t find_target(
    const RawDataset& data,
    const std::string& target_name
) {
    const auto it = std::find(
        data.column_names.begin(),
        data.column_names.end(),
        target_name
    );

    if (it == data.column_names.end()) {
        throw std::invalid_argument(
            "Target column not found: " + target_name
        );
    }

    return static_cast<std::size_t>(
        std::distance(data.column_names.begin(), it)
    );
}

Dataset make_legacy_dataset(
    const RawDataset& raw,
    const std::vector<FeatureType>& types,
    const std::string& target_name
) {
    Dataset result = make_typed_dataset(raw, types);

    const std::size_t target_index =
        find_target(raw, target_name);

    // Existing preprocessing identifies the target by name.
    // Rename it only in this temporary typed Dataset.
    // Original RawDataset and its schema remain unchanged.
    if (target_name != legacy_target_name) {
        for (std::size_t i = 0;
             i < result.column_names.size();
             ++i) {
            if (i != target_index &&
                result.column_names[i] == legacy_target_name) {
                throw std::invalid_argument(
                    "Feature name conflicts with legacy target"
                );
            }
        }
    }

    result.column_names[target_index] = legacy_target_name;

    return result;
}

std::string read_scalar(
    const YAML::Node& parent,
    const std::string& key
) {
    if (!parent.IsMap()) {
        throw std::invalid_argument(
            "Expected YAML mapping containing: " + key
        );
    }

    const YAML::Node value = parent[key];

    if (!value.IsDefined() || !value.IsScalar()) {
        throw std::invalid_argument(
            "Missing or invalid YAML option: " + key
        );
    }

    return value.as<std::string>();
}

} // namespace

FittedRawPreprocessing fit_raw_preprocessing(
    const RawDataset& train,
    const std::string& target_name,
    const PreprocessingOptions& options
) {
    validate_raw(train, false);

    const std::size_t target_index =
        find_target(train, target_name);

    // A target must never contain missing labels.
    for (const auto& value : train.columns[target_index]) {
        if (!value.has_value()) {
            throw std::invalid_argument(
                "Target contains missing values"
            );
        }
    }

    FittedRawPreprocessing fitted;

    fitted.column_names = train.column_names;
    fitted.target_name = target_name;
    fitted.column_types.reserve(train.columns.size());

    for (std::size_t i = 0;
         i < train.columns.size();
         ++i) {

        if (i == target_index) {
            // Keep the original target text unchanged.
            // Numeric regression targets are converted later,
            // outside the feature preprocessing.
            fitted.column_types.push_back(
                FeatureType::Categorical
            );
        } else {
            // Feature types are determined once, on train.
            fitted.column_types.push_back(
                detect_feature_type(train.columns[i])
            );
        }
    }

    Dataset typed_train = make_legacy_dataset(
        train,
        fitted.column_types,
        target_name
    );

    // Reuse the existing Lab 1 preprocessing algorithm.
    fitted.state = fit_preprocessing(
        typed_train,
        options
    );

    return fitted;
}

NumericMatrix transform_raw_preprocessing(
    const RawDataset& data,
    const FittedRawPreprocessing& fitted
) {
    validate_raw(data, true);

    if (data.column_names != fitted.column_names) {
        throw std::invalid_argument(
            "Dataset schema differs from fitted train schema"
        );
    }

    if (fitted.column_types.size() != data.columns.size()) {
        throw std::invalid_argument(
            "Invalid fitted feature type metadata"
        );
    }

    Dataset typed_data = make_legacy_dataset(
        data,
        fitted.column_types,
        fitted.target_name
    );

    // No calls to detect_feature_type() or
    // fit_preprocessing() are allowed here.
    return transform_preprocessing(
        typed_data,
        fitted.state
    );
}

PreprocessingOptions load_preprocessing_config(
    const std::string& filename
) {
    const YAML::Node root = YAML::LoadFile(filename);

    if (!root.IsMap()) {
        throw std::invalid_argument(
            "Preprocessing configuration must be a YAML map"
        );
    }

    const YAML::Node numeric = root["numeric"];
    const YAML::Node categorical = root["categorical"];

    if (!numeric.IsMap() || !categorical.IsMap()) {
        throw std::invalid_argument(
            "Expected numeric and categorical sections"
        );
    }

    PreprocessingOptions options;

    const std::string imputation =
        read_scalar(numeric, "imputation");

    if (imputation == "mean") {
        options.imputation = NumericImputation::Mean;
    } else if (imputation == "median") {
        options.imputation = NumericImputation::Median;
    } else if (imputation == "zero") {
        options.imputation = NumericImputation::Zero;
    } else {
        throw std::invalid_argument(
            "Unsupported numeric imputation: " + imputation
        );
    }

    const std::string scaling =
        read_scalar(numeric, "scaling");

    if (scaling == "minmax") {
        options.scaling = NumericScaling::MinMax;
    } else if (scaling == "standard") {
        options.scaling = NumericScaling::Standard;
    } else if (scaling == "robust") {
        options.scaling = NumericScaling::Robust;
    } else {
        throw std::invalid_argument(
            "Unsupported numeric scaling: " + scaling
        );
    }

    const std::string categorical_imputation =
        read_scalar(categorical, "imputation");

    // The existing preprocessing uses a dedicated
    // "__MISSING__" category for absent values.
    if (categorical_imputation != "missing_category") {
        throw std::invalid_argument(
            "Supported categorical imputation: missing_category"
        );
    }

    const std::string encoding =
        read_scalar(categorical, "encoding");

    if (encoding == "one_hot") {
        options.encoding = CategoryEncoding::OneHot;
    } else if (encoding == "ordinal") {
        options.encoding = CategoryEncoding::Ordinal;
    } else {
        throw std::invalid_argument(
            "Unsupported category encoding: " + encoding
        );
    }

    return options;
}

} // namespace lab5




