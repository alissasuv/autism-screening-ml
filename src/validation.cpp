#include "validation.hpp"

#include "lab5_preprocessing.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace lab5 {
namespace {

std::size_t target_column(
    const RawDataset& data,
    const std::string& name
) {
    if (data.row_count == 0 ||
        data.column_names.size() != data.columns.size()) {
        throw std::invalid_argument(
            "Invalid raw training dataset"
        );
    }

    std::size_t found = data.column_names.size();

    for (std::size_t i = 0; i < data.column_names.size(); ++i) {
        if (data.columns[i].size() != data.row_count) {
            throw std::invalid_argument(
                "Inconsistent raw column size"
            );
        }

        if (data.column_names[i] == name) {
            if (found != data.column_names.size()) {
                throw std::invalid_argument(
                    "Duplicate target column"
                );
            }
            found = i;
        }
    }

    if (found == data.column_names.size()) {
        throw std::invalid_argument(
            "Target column not found: " + name
        );
    }

    return found;
}

std::vector<std::string> target_text(
    const RawDataset& data,
    const std::string& name
) {
    const auto& column =
        data.columns[target_column(data, name)];

    std::vector<std::string> values;
    values.reserve(data.row_count);

    for (const auto& cell : column) {
        if (!cell.has_value()) {
            throw std::invalid_argument(
                "Missing target value"
            );
        }
        values.push_back(*cell);
    }

    return values;
}

double parse_finite(const std::string& text) {
    double result = 0.0;

    const auto [end, ec] = std::from_chars(
        text.data(),
        text.data() + text.size(),
        result
    );

    if (ec != std::errc{} ||
        end != text.data() + text.size() ||
        !std::isfinite(result)) {
        throw std::invalid_argument(
            "Invalid numeric regression target: " + text
        );
    }

    return result;
}

std::vector<double> numerical_targets(
    const std::vector<std::string>& text
) {
    std::vector<double> values;
    values.reserve(text.size());

    for (const auto& value : text) {
        values.push_back(parse_finite(value));
    }

    return values;
}

std::vector<double> encoded_labels(
    const std::vector<std::string>& text,
    const std::map<std::string, double>& mapping
) {
    std::vector<double> values;
    values.reserve(text.size());

    for (const auto& label : text) {
        const auto it = mapping.find(label);

        if (it == mapping.end()) {
            throw std::invalid_argument(
                "Unknown class: " + label
            );
        }

        values.push_back(it->second);
    }

    return values;
}

void validate_folds(
    const std::vector<Fold>& folds,
    std::size_t rows
) {
    if (folds.size() < 2 || folds.size() > rows) {
        throw std::invalid_argument(
            "Invalid number of CV folds"
        );
    }

    std::vector<std::size_t> coverage(rows, 0);

    for (const auto& fold : folds) {
        if (fold.train_indices.empty() ||
            fold.validation_indices.empty()) {
            throw std::invalid_argument(
                "Empty CV train or validation split"
            );
        }

        std::vector<bool> in_train(rows, false);

        for (const auto index : fold.train_indices) {
            if (index >= rows || in_train[index]) {
                throw std::invalid_argument(
                    "Invalid/duplicate CV training index"
                );
            }

            in_train[index] = true;
        }

        std::vector<bool> in_valid(rows, false);

        for (const auto index : fold.validation_indices) {
            if (index >= rows ||
                in_train[index] ||
                in_valid[index]) {
                throw std::invalid_argument(
                    "Overlapping or duplicate CV indices"
                );
            }

            in_valid[index] = true;
            ++coverage[index];
        }

        if (fold.train_indices.size() +
                fold.validation_indices.size() != rows) {
            throw std::invalid_argument(
                "A fold does not partition the training set"
            );
        }
    }

    for (const auto count : coverage) {
        if (count != 1) {
            throw std::invalid_argument(
                "CV validation folds must cover each row once"
            );
        }
    }
}

} // namespace

RawDataset select_raw_rows(
    const RawDataset& data,
    const std::vector<std::size_t>& indices
) {
    if (data.columns.size() != data.column_names.size()) {
        throw std::invalid_argument(
            "Invalid raw schema"
        );
    }

    RawDataset result;
    result.row_count = indices.size();
    result.column_names = data.column_names;
    result.columns.resize(data.columns.size());

    for (std::size_t c = 0; c < data.columns.size(); ++c) {
        if (data.columns[c].size() != data.row_count) {
            throw std::invalid_argument(
                "Invalid raw column length"
            );
        }

        result.columns[c].reserve(indices.size());

        for (const auto i : indices) {
            if (i >= data.row_count) {
                throw std::out_of_range(
                    "Raw row index out of range"
                );
            }

            result.columns[c].push_back(
                data.columns[c][i]
            );
        }
    }

    return result;
}

std::vector<Fold> make_folds(
    const RawDataset& data,
    const std::string& target_name,
    Task task,
    std::size_t n_splits,
    std::uint32_t random_state
) {
    const auto labels = target_text(data, target_name);

    if (n_splits < 2 || n_splits > data.row_count) {
        throw std::invalid_argument(
            "n_splits must be between 2 and n_train"
        );
    }

    std::mt19937 rng(random_state);
    std::vector<std::vector<std::size_t>> validation(n_splits);

    if (task == Task::Classification) {
        std::map<
            std::string,
            std::vector<std::size_t>
        > groups;

        for (std::size_t i = 0; i < labels.size(); ++i) {
            groups[labels[i]].push_back(i);
        }

        std::size_t offset = 0;

        for (auto& [label, indices] : groups) {
            (void)label;

            if (indices.size() < n_splits) {
                throw std::invalid_argument(
                    "Each class must contain at least n_splits rows"
                );
            }

            std::shuffle(
                indices.begin(),
                indices.end(),
                rng
            );

            for (std::size_t i = 0; i < indices.size(); ++i) {
                validation[(offset + i) % n_splits].push_back(
                    indices[i]
                );
            }

            offset = (offset + indices.size()) % n_splits;
        }
    } else {
        // Regression requires numeric target values.
        (void)numerical_targets(labels);

        std::vector<std::size_t> indices(data.row_count);
        std::iota(indices.begin(), indices.end(), 0);

        std::shuffle(
            indices.begin(),
            indices.end(),
            rng
        );

        for (std::size_t i = 0; i < indices.size(); ++i) {
            validation[i % n_splits].push_back(indices[i]);
        }
    }

    std::vector<Fold> folds(n_splits);

    for (std::size_t f = 0; f < n_splits; ++f) {
        auto& fold = folds[f];

        fold.validation_indices = std::move(validation[f]);

        std::sort(
            fold.validation_indices.begin(),
            fold.validation_indices.end()
        );

        std::vector<bool> selected(data.row_count, false);

        for (const auto i : fold.validation_indices) {
            selected[i] = true;
        }

        for (std::size_t i = 0; i < data.row_count; ++i) {
            if (!selected[i]) {
                fold.train_indices.push_back(i);
            }
        }
    }

    validate_folds(folds, data.row_count);
    return folds;
}

CrossValidationResult cross_validate(
    const RawDataset& train,
    const std::string& target_name,
    const PreprocessingOptions& preprocessing,
    const ModelConfig& model,
    Scoring scoring,
    const std::vector<Fold>& folds
) {
    validate_scoring(
        scoring,
        model.task == Task::Classification
    );

    (void)target_column(train, target_name);
    validate_folds(folds, train.row_count);

    std::map<std::string, double> mapping;
    std::vector<double> classes;

    if (model.task == Task::Classification) {
        // Fix class IDs across all folds.
        const auto labels = target_text(train, target_name);

        for (const auto& label : labels) {
            mapping.emplace(label, 0.0);
        }

        for (auto& [label, value] : mapping) {
            (void)label;

            value = static_cast<double>(classes.size());
            classes.push_back(value);
        }
    }

    CrossValidationResult result;
    result.fold_scores.reserve(folds.size());

    for (const auto& fold : folds) {
        RawDataset fold_train = select_raw_rows(
            train,
            fold.train_indices
        );

        RawDataset fold_valid = select_raw_rows(
            train,
            fold.validation_indices
        );

        // Fit preprocessing exclusively on this fold's train.
        const auto fitted = fit_raw_preprocessing(
            fold_train,
            target_name,
            preprocessing
        );

        const NumericMatrix x_train =
            transform_raw_preprocessing(
                fold_train,
                fitted
            );

        const NumericMatrix x_valid =
            transform_raw_preprocessing(
                fold_valid,
                fitted
            );

        if (x_train.cols == 0) {
            throw std::invalid_argument(
                "No features after preprocessing"
            );
        }

        double score = 0.0;

        if (model.task == Task::Classification) {
            const auto y_train = encoded_labels(
                x_train.target_values,
                mapping
            );

            const auto y_valid = encoded_labels(
                x_valid.target_values,
                mapping
            );

            auto estimator = create_classifier(model);

            estimator->fit(x_train, y_train);

            const auto predictions =
                estimator->predict(x_valid);

            score = classification_scoring(
                scoring,
                y_valid,
                predictions,
                classes
            );
        } else {
            const auto y_train = numerical_targets(
                x_train.target_values
            );

            const auto y_valid = numerical_targets(
                x_valid.target_values
            );

            auto estimator = create_regressor(model);

            estimator->fit(x_train, y_train);

            const auto predictions =
                estimator->predict(x_valid);

            score = regression_scoring(
                scoring,
                y_valid,
                predictions
            );
        }

        if (!std::isfinite(score)) {
            throw std::runtime_error(
                "Non-finite cross-validation score"
            );
        }

        result.fold_scores.push_back(score);
    }

    const double sum = std::accumulate(
        result.fold_scores.begin(),
        result.fold_scores.end(),
        0.0
    );

    result.mean =
        sum / static_cast<double>(result.fold_scores.size());

    double squared = 0.0;

    for (const auto score : result.fold_scores) {
        const double delta = score - result.mean;
        squared += delta * delta;
    }

    result.stddev = std::sqrt(
        squared / static_cast<double>(result.fold_scores.size())
    );

    return result;
}

} // namespace lab5

