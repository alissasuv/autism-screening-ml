#include "decision_stump.hpp"
#include "ml_exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

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
        double sum = 0.0;

        for (const auto index : indices) {
            sum += target[index];
        }

        return sum / static_cast<double>(indices.size());
    }

    std::map<double, std::size_t> counts;

    for (const auto index : indices) {
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

    const std::size_t n = features.rows;

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);

    constant_prediction_ = calculate_prediction(target, order);
    left_prediction_ = constant_prediction_;
    right_prediction_ = constant_prediction_;

    double total_sum = 0.0;
    std::map<double, std::size_t> class_counts;

    for (const double value : target) {
        if (criterion_ == SplitCriterion::Variance) {
            total_sum += value;
        } else {
            ++class_counts[value];
        }
    }

    std::map<double, std::size_t> class_ids;
    std::size_t next_id = 0;

    for (const auto& [label, count] : class_counts) {
        (void)count;
        class_ids[label] = next_id++;
    }

    std::vector<std::size_t> encoded_target(n, 0);
    std::vector<std::size_t> total_counts(class_ids.size(), 0);

    if (criterion_ == SplitCriterion::Gini) {
        for (std::size_t i = 0; i < n; ++i) {
            const auto id = class_ids.at(target[i]);
            encoded_target[i] = id;
            ++total_counts[id];
        }
    }

    double best_score = -std::numeric_limits<double>::infinity();

    for (std::size_t feature = 0;
         feature < feature_count_;
         ++feature) {

        std::iota(order.begin(), order.end(), 0);

        std::sort(
            order.begin(),
            order.end(),
            [&](std::size_t a, std::size_t b) {
                if (features[a][feature] != features[b][feature]) {
                    return features[a][feature] < features[b][feature];
                }
                return a < b;
            }
        );

        double left_sum = 0.0;
        std::size_t left_size = 0;

        std::vector<std::size_t> left_counts(
            total_counts.size(), 0
        );
        std::vector<std::size_t> right_counts = total_counts;

        double left_squared_sum = 0.0;
        double right_squared_sum = 0.0;

        if (criterion_ == SplitCriterion::Gini) {
            for (const auto count : right_counts) {
                right_squared_sum +=
                    static_cast<double>(count) * count;
            }
        }

        for (std::size_t p = 0; p + 1 < n; ++p) {
            const std::size_t index = order[p];

            ++left_size;
            const std::size_t right_size = n - left_size;

            if (criterion_ == SplitCriterion::Variance) {
                left_sum += target[index];
            } else {
                const auto label = encoded_target[index];

                const double a =
                    static_cast<double>(left_counts[label]);
                const double b =
                    static_cast<double>(right_counts[label]);

                left_squared_sum += 2.0 * a + 1.0;
                right_squared_sum += 1.0 - 2.0 * b;

                ++left_counts[label];
                --right_counts[label];
            }

            const double current = features[index][feature];
            const double next = features[order[p + 1]][feature];

            if (current == next) {
                continue;
            }

            if (left_size < min_samples_leaf_ ||
                right_size < min_samples_leaf_) {
                continue;
            }

            double score = 0.0;

            if (criterion_ == SplitCriterion::Variance) {
                const double right_sum =
                    total_sum - left_sum;

                score =
                    left_sum * left_sum / left_size +
                    right_sum * right_sum / right_size;
            } else {
                score =
                    left_squared_sum / left_size +
                    right_squared_sum / right_size;
            }

            const double threshold =
                std::midpoint(current, next);

            if (score > best_score) {
                best_score = score;
                split_feature_ = feature;
                split_threshold_ = threshold;
                has_split_ = true;
            }
        }
    }

    if (has_split_) {
        std::vector<std::size_t> left_indices;
        std::vector<std::size_t> right_indices;

        for (std::size_t i = 0; i < n; ++i) {
            if (features[i][split_feature_] <= split_threshold_) {
                left_indices.push_back(i);
            } else {
                right_indices.push_back(i);
            }
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
        } else if (features[i][split_feature_] <= split_threshold_) {
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
