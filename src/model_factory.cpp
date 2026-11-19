#include "model_factory.hpp"

#include <stdexcept>

namespace lab5 {

std::unique_ptr<IClassifier> create_classifier(
    const ModelConfig& config
) {
    if (config.task != Task::Classification) {
        throw std::invalid_argument(
            "Expected classification model configuration"
        );
    }

    return std::make_unique<RandomForestClassifier>(
        config.parameters
    );
}

std::unique_ptr<IRegressor> create_regressor(
    const ModelConfig& config
) {
    if (config.task != Task::Regression) {
        throw std::invalid_argument(
            "Expected regression model configuration"
        );
    }

    return std::make_unique<RandomForestRegressor>(
        config.parameters
    );
}

} // namespace lab5


