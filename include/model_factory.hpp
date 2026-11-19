#pragma once

#include "random_forest.hpp"

#include <memory>

namespace lab5 {

enum class Task {
    Classification,
    Regression
};

struct ModelConfig {
    Task task = Task::Classification;
    RandomForestParameters parameters =
        classifier_default_parameters();
};

std::unique_ptr<IClassifier> create_classifier(
    const ModelConfig& config
);

std::unique_ptr<IRegressor> create_regressor(
    const ModelConfig& config
);

} // namespace lab5



