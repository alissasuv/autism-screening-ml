#include "feature_type.hpp"

#include <charconv>
#include <string>
#include <system_error>
#include <unordered_set>

namespace {

bool is_number(const std::string& value) {
    if (value.empty()) {
        return false;
    }

    double number = 0.0;

    const char* begin = value.data();
    const char* end = begin + value.size();

    auto [ptr, error] = std::from_chars(begin, end, number);

    return error == std::errc{} && ptr == end;
}

} // namespace

FeatureType detect_feature_type(
    const std::vector<std::optional<std::string>>& values
) {
    std::unordered_set<std::string> unique_values;

    for (const auto& value : values) {
        if (!value.has_value()) {
            continue;
        }

        if (!is_number(*value)) {
            return FeatureType::Categorical;
        }

        unique_values.insert(*value);
    }

    if (unique_values.size() <= 2) {
        return FeatureType::Categorical;
    }

    return FeatureType::Numerical;
}