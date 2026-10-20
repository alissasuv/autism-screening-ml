#pragma once

#include <optional>
#include <string>
#include <vector>

enum class FeatureType {
    Numerical,
    Categorical
};

FeatureType detect_feature_type(
    const std::vector<std::optional<std::string>>& values
);