
#include "decision_stump.hpp"
#include "ml_exceptions.hpp"

#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

DecisionStump::DecisionStump(
    SplitCriterion criterion,
    std::size_t min_samples_leaf
)
    : criterion_(criterion),
      min_samples_leaf_(min_samples_leaf) {
    if (min_samples_leaf_ == 0) {
        throw InvalidModelParameterError(
            "min_samples_leaf must be at least 1"
        );
    }
}

double DecisionStump::calculate_prediction(
    std::span<const double> target,
    const std::vector<std::size_t>& indices
) const {
    if (indices.empty()) {
        throw InvalidModelStateError(
            "Cannot predict from an empty leaf"
        );
    }

    if (criterion_ == SplitCriterion::Variance) {
        long double sum = 0.0L;

        for (const std::size_t index : indices) {
            sum += static_cast<long double>(target[index]);
        }

        return static_cast<double>(
            sum / static_cast<long double>(indices.size())
        );
    }

    std::map<double, std::size_t> counts;

    for (const std::size_t index : indices) {
        ++counts[target[index]];
    }

    double best_class = 0.0;
    std::size_t best_count = 0;

    for (const auto& [label, count] : counts) {
        if (count > best_count) {
            best_count = count;
            best_class = label;
        }
    }

    return best_class;
}

void DecisionStump::fit(
    const NumericMatrix& features,
    std::span<const double> target
) {
    fitted_ = false;
    has_split_ = false;

    if (features.rows == 0 || features.cols == 0) {
        throw InvalidModelParameterError(
            "Training features must not be empty"
        );
    }

    if (features.rows != target.size()) {
        throw DimensionMismatchError(
            "Features and target sizes do not match"
        );
    }

    if (features.rows >
        std::numeric_limits<std::size_t>::max() / features.cols) {
        throw DimensionMismatchError(
            "Feature matrix dimensions overflow"
        );
    }

    if (features.values.size() != features.rows * features.cols) {
        throw DimensionMismatchError(
            "Invalid feature matrix dimensions"
        );
    }

    for (const double value : target) {
        if (!std::isfinite(value)) {
            throw InvalidModelParameterError(
                "Target contains a non-finite value"
            );
        }
    }

    for (const double value : features.values) {
        if (!std::isfinite(value)) {
            throw InvalidModelParameterError(
                "Features contain a non-finite value"
            );
        }
    }

    feature_count_ = features.cols;

    std::vector<std::size_t> object_ids(features.rows);
    std::iota(object_ids.begin(), object_ids.end(), 0);

    std::vector<std::size_t> feature_ids(features.cols);
    std::iota(feature_ids.begin(), feature_ids.end(), 0);

    constant_prediction_ =
        calculate_prediction(target, object_ids);

    left_prediction_ = constant_prediction_;
    right_prediction_ = constant_prediction_;

    const auto best = split_finder_.find_best(
        features,
        target,
        object_ids,
        feature_ids,
        criterion_,
        min_samples_leaf_
    );

    if (best.has_value()) {
        split_feature_ = best->feature;
        split_threshold_ = best->threshold;
        has_split_ = true;

        std::vector<std::size_t> left_indices;
        std::vector<std::size_t> right_indices;

        left_indices.reserve(features.rows);
        right_indices.reserve(features.rows);

        for (const std::size_t index : object_ids) {
            if (features(index, split_feature_) <= split_threshold_) {
                left_indices.push_back(index);
            } else {
                right_indices.push_back(index);
            }
        }

        if (left_indices.empty() || right_indices.empty()) {
            throw InvalidModelStateError(
                "Split produced an empty leaf"
            );
        }

        left_prediction_ =
            calculate_prediction(target, left_indices);

        right_prediction_ =
            calculate_prediction(target, right_indices);
    }

    fitted_ = true;
}

std::vector<double> DecisionStump::predict(
    const NumericMatrix& features
) const {
    if (!fitted_) {
        throw InvalidModelStateError(
            "Model must be fitted before prediction"
        );
    }

    if (features.cols != feature_count_) {
        throw DimensionMismatchError(
            "Prediction feature count does not match training"
        );
    }

    if (features.cols != 0 &&
        features.rows >
            std::numeric_limits<std::size_t>::max() / features.cols) {
        throw DimensionMismatchError(
            "Prediction matrix dimensions overflow"
        );
    }

    if (features.values.size() != features.rows * features.cols) {
        throw DimensionMismatchError(
            "Invalid prediction matrix dimensions"
        );
    }

    std::vector<double> predictions;
    predictions.reserve(features.rows);

    for (std::size_t i = 0; i < features.rows; ++i) {
        if (!has_split_) {
            predictions.push_back(constant_prediction_);
            continue;
        }

        const double value = features(i, split_feature_);

        if (!std::isfinite(value)) {
            throw InvalidModelParameterError(
                "Prediction features contain a non-finite value"
            );
        }

        if (value <= split_threshold_) {
            predictions.push_back(left_prediction_);
        } else {
            predictions.push_back(right_prediction_);
        }
    }

    return predictions;
}

std::size_t DecisionStump::split_feature() const {
    if (!fitted_ || !has_split_) {
        throw InvalidModelStateError(
            "Model has no fitted split"
        );
    }

    return split_feature_;
}

double DecisionStump::split_threshold() const {
    if (!fitted_ || !has_split_) {
        throw InvalidModelStateError(
            "Model has no fitted split"
        );
    }

    return split_threshold_;
}

double DecisionStump::left_prediction() const {
    if (!fitted_) {
        throw InvalidModelStateError("Model is not fitted");
    }

    return left_prediction_;
}

double DecisionStump::right_prediction() const {
    if (!fitted_) {
        throw InvalidModelStateError("Model is not fitted");
    }

    return right_prediction_;
}

bool DecisionStump::has_split() const {
    if (!fitted_) {
        throw InvalidModelStateError("Model is not fitted");
    }

    return has_split_;
}
