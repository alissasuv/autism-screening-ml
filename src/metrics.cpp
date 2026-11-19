#include "metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab5 {
namespace {

void validate_predictions(
    std::span<const double> actual,
    std::span<const double> predicted
) {
    if (actual.empty()) {
        throw std::invalid_argument(
            "Metric input must not be empty"
        );
    }

    if (actual.size() != predicted.size()) {
        throw std::invalid_argument(
            "Actual and predicted sizes do not match"
        );
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) ||
            !std::isfinite(predicted[i])) {
            throw std::invalid_argument(
                "Metrics require finite numeric values"
            );
        }
    }
}

double checked_double(long double value) {
    const double result = static_cast<double>(value);

    if (!std::isfinite(result)) {
        throw std::overflow_error(
            "Metric result is not finite"
        );
    }

    return result;
}

void validate_classes(std::span<const double> classes) {
    if (classes.empty()) {
        throw std::invalid_argument(
            "Classification requires at least one class"
        );
    }

    for (std::size_t i = 0; i < classes.size(); ++i) {
        if (!std::isfinite(classes[i])) {
            throw std::invalid_argument(
                "Class labels must be finite"
            );
        }

        for (std::size_t j = 0; j < i; ++j) {
            if (classes[i] == classes[j]) {
                throw std::invalid_argument(
                    "Duplicate class label"
                );
            }
        }
    }
}

std::size_t class_index(
    double label,
    std::span<const double> classes
) {
    const auto it = std::find(
        classes.begin(),
        classes.end(),
        label
    );

    if (it == classes.end()) {
        throw std::invalid_argument(
            "Label is absent from the fixed class list"
        );
    }

    return static_cast<std::size_t>(
        std::distance(classes.begin(), it)
    );
}

double safe_ratio(
    std::size_t numerator,
    std::size_t denominator
) {
    if (denominator == 0) {
        return 0.0;
    }

    return static_cast<double>(numerator) /
           static_cast<double>(denominator);
}

} // namespace

Scoring parse_scoring(const std::string& name) {
    if (name == "mse") {
        return Scoring::MSE;
    }
    if (name == "mae") {
        return Scoring::MAE;
    }
    if (name == "r2") {
        return Scoring::R2;
    }
    if (name == "accuracy") {
        return Scoring::Accuracy;
    }
    if (name == "precision") {
        return Scoring::Precision;
    }
    if (name == "recall") {
        return Scoring::Recall;
    }
    if (name == "f1") {
        return Scoring::F1;
    }

    throw std::invalid_argument(
        "Unsupported scoring metric: " + name
    );
}

std::string scoring_name(Scoring scoring) {
    switch (scoring) {
        case Scoring::MSE:
            return "mse";
        case Scoring::MAE:
            return "mae";
        case Scoring::R2:
            return "r2";
        case Scoring::Accuracy:
            return "accuracy";
        case Scoring::Precision:
            return "precision";
        case Scoring::Recall:
            return "recall";
        case Scoring::F1:
            return "f1";
    }

    throw std::invalid_argument(
        "Unknown scoring enum value"
    );
}

bool higher_is_better(Scoring scoring) {
    switch (scoring) {
        case Scoring::MSE:
        case Scoring::MAE:
            return false;

        case Scoring::R2:
        case Scoring::Accuracy:
        case Scoring::Precision:
        case Scoring::Recall:
        case Scoring::F1:
            return true;
    }

    throw std::invalid_argument(
        "Unknown scoring enum value"
    );
}

void validate_scoring(
    Scoring scoring,
    bool classification
) {
    const bool regression_metric =
        scoring == Scoring::MSE ||
        scoring == Scoring::MAE ||
        scoring == Scoring::R2;

    if (classification == regression_metric) {
        throw std::invalid_argument(
            "Scoring metric does not match the task"
        );
    }
}

double mean_squared_error(
    std::span<const double> actual,
    std::span<const double> predicted
) {
    validate_predictions(actual, predicted);

    long double sum = 0.0L;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const long double difference =
            static_cast<long double>(actual[i]) -
            static_cast<long double>(predicted[i]);

        sum += difference * difference;
    }

    return checked_double(
        sum / static_cast<long double>(actual.size())
    );
}

double mean_absolute_error(
    std::span<const double> actual,
    std::span<const double> predicted
) {
    validate_predictions(actual, predicted);

    long double sum = 0.0L;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const long double difference =
            static_cast<long double>(actual[i]) -
            static_cast<long double>(predicted[i]);

        sum += std::abs(difference);
    }

    return checked_double(
        sum / static_cast<long double>(actual.size())
    );
}

double r2_score(
    std::span<const double> actual,
    std::span<const double> predicted
) {
    validate_predictions(actual, predicted);

    if (actual.size() < 2) {
        throw std::invalid_argument(
            "R2 requires at least two objects"
        );
    }

    long double sum = 0.0L;

    for (const double value : actual) {
        sum += static_cast<long double>(value);
    }

    const long double mean =
        sum / static_cast<long double>(actual.size());

    long double residual_sum = 0.0L;
    long double total_sum = 0.0L;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const long double residual =
            static_cast<long double>(actual[i]) -
            static_cast<long double>(predicted[i]);

        const long double deviation =
            static_cast<long double>(actual[i]) - mean;

        residual_sum += residual * residual;
        total_sum += deviation * deviation;
    }

    // Coursework convention for constant targets.
    if (total_sum == 0.0L) {
        return residual_sum == 0.0L ? 1.0 : 0.0;
    }

    return checked_double(
        1.0L - residual_sum / total_sum
    );
}

RegressionMetrics regression_metrics(
    std::span<const double> actual,
    std::span<const double> predicted
) {
    RegressionMetrics result;

    result.mse = mean_squared_error(actual, predicted);
    result.mae = mean_absolute_error(actual, predicted);
    result.r2 = r2_score(actual, predicted);

    return result;
}

ClassificationMetrics classification_metrics(
    std::span<const double> actual,
    std::span<const double> predicted,
    std::span<const double> classes
) {
    validate_predictions(actual, predicted);
    validate_classes(classes);

    const std::size_t class_count = classes.size();

    std::vector<std::size_t> true_positives(class_count, 0);
    std::vector<std::size_t> actual_counts(class_count, 0);
    std::vector<std::size_t> predicted_counts(class_count, 0);

    std::size_t correct = 0;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const std::size_t actual_index =
            class_index(actual[i], classes);

        const std::size_t predicted_index =
            class_index(predicted[i], classes);

        ++actual_counts[actual_index];
        ++predicted_counts[predicted_index];

        if (actual_index == predicted_index) {
            ++correct;
            ++true_positives[actual_index];
        }
    }

    ClassificationMetrics result;

    result.accuracy = safe_ratio(
        correct,
        actual.size()
    );

    for (std::size_t c = 0; c < class_count; ++c) {
        const double precision = safe_ratio(
            true_positives[c],
            predicted_counts[c]
        );

        const double recall = safe_ratio(
            true_positives[c],
            actual_counts[c]
        );

        double f1 = 0.0;

        if (precision + recall > 0.0) {
            f1 = 2.0 * precision * recall /
                 (precision + recall);
        }

        result.precision += precision;
        result.recall += recall;
        result.f1 += f1;
    }

    const double divisor =
        static_cast<double>(class_count);

    result.precision /= divisor;
    result.recall /= divisor;
    result.f1 /= divisor;

    return result;
}

double regression_scoring(
    Scoring scoring,
    std::span<const double> actual,
    std::span<const double> predicted
) {
    validate_scoring(scoring, false);

    switch (scoring) {
        case Scoring::MSE:
            return mean_squared_error(actual, predicted);

        case Scoring::MAE:
            return mean_absolute_error(actual, predicted);

        case Scoring::R2:
            return r2_score(actual, predicted);

        default:
            break;
    }

    throw std::invalid_argument(
        "Unsupported regression scoring"
    );
}

double classification_scoring(
    Scoring scoring,
    std::span<const double> actual,
    std::span<const double> predicted,
    std::span<const double> classes
) {
    validate_scoring(scoring, true);

    const ClassificationMetrics result =
        classification_metrics(actual, predicted, classes);

    switch (scoring) {
        case Scoring::Accuracy:
            return result.accuracy;

        case Scoring::Precision:
            return result.precision;

        case Scoring::Recall:
            return result.recall;

        case Scoring::F1:
            return result.f1;

        default:
            break;
    }

    throw std::invalid_argument(
        "Unsupported classification scoring"
    );
}

bool is_better_score(
    double candidate,
    double current_best,
    Scoring scoring
) {
    if (!std::isfinite(candidate) ||
        !std::isfinite(current_best)) {
        throw std::invalid_argument(
            "Cannot compare non-finite metric scores"
        );
    }

    // Strict comparison preserves the first candidate on ties.
    return higher_is_better(scoring)
        ? candidate > current_best
        : candidate < current_best;
}

} // namespace lab5


