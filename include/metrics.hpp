#pragma once

#include <span>
#include <string>
#include <vector>

namespace lab5 {

enum class Scoring {
    MSE,
    MAE,
    R2,
    Accuracy,
    Precision,
    Recall,
    F1
};

struct RegressionMetrics {
    double mse = 0.0;
    double mae = 0.0;
    double r2 = 0.0;
};

struct ClassificationMetrics {
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
};

// Parse one of the supported scoring names.
Scoring parse_scoring(const std::string& name);

// Return the canonical name used in configuration files.
std::string scoring_name(Scoring scoring);

// Return true for metrics that must be maximized.
bool higher_is_better(Scoring scoring);

// Check whether a scoring metric matches the task.
void validate_scoring(
    Scoring scoring,
    bool classification
);

// Regression metrics.
double mean_squared_error(
    std::span<const double> actual,
    std::span<const double> predicted
);

double mean_absolute_error(
    std::span<const double> actual,
    std::span<const double> predicted
);

double r2_score(
    std::span<const double> actual,
    std::span<const double> predicted
);

RegressionMetrics regression_metrics(
    std::span<const double> actual,
    std::span<const double> predicted
);

// Classification metrics.
// 'classes' is the complete, fixed list of class labels.
ClassificationMetrics classification_metrics(
    std::span<const double> actual,
    std::span<const double> predicted,
    std::span<const double> classes
);

// Return one scoring value using the shared metric definitions.
double regression_scoring(
    Scoring scoring,
    std::span<const double> actual,
    std::span<const double> predicted
);

double classification_scoring(
    Scoring scoring,
    std::span<const double> actual,
    std::span<const double> predicted,
    std::span<const double> classes
);

// Compare candidate scores according to metric direction.
bool is_better_score(
    double candidate,
    double current_best,
    Scoring scoring
);

} // namespace lab5

