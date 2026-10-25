#include "feature_type.hpp"

#include <charconv>
#include <cmath>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <vector>

namespace {

bool parse_number(
    const std::string& text,
    double& result
) {
    if (text.empty()) {
        return false;
    }

    const char* begin = text.data();
    const char* end = begin + text.size();

    const auto [ptr, error] =
        std::from_chars(begin, end, result);

    return error == std::errc{} &&
           ptr == end &&
           std::isfinite(result);
}

} // namespace

FeatureType detect_feature_type(
    const std::vector<std::optional<std::string>>& values
) {
    std::set<double> unique_numbers;
    bool has_values = false;

    for (const auto& cell : values) {
        if (!cell.has_value()) {
            continue;
        }

        has_values = true;

        double number = 0.0;

        // Any non-numeric value makes the column categorical.
        if (!parse_number(*cell, number)) {
            return FeatureType::Categorical;
        }

        unique_numbers.insert(number);
    }

    // A completely missing column cannot be treated as numeric.
    if (!has_values) {
        return FeatureType::Categorical;
    }

    // Binary and constant numerical columns are categorical.
    if (unique_numbers.size() <= 2) {
        return FeatureType::Categorical;
    }

    return FeatureType::Numerical;
}
