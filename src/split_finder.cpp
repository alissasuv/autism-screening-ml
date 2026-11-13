
#include "split_finder.hpp"
#include "ml_exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

namespace {

void validate_request(
    const NumericMatrix& features,
    std::span<const double> target,
    std::span<const std::size_t> object_ids,
    std::span<const std::size_t> feature_ids,
    std::size_t min_samples_leaf,
    std::span<const std::size_t> counts
) {
    if (min_samples_leaf == 0) {
        throw InvalidModelParameterError(
            "min_samples_leaf must be at least 1"
        );
    }

    if (features.rows != target.size()) {
        throw DimensionMismatchError(
            "Features and target sizes do not match"
        );
    }

    if (features.cols != 0 &&
        features.rows >
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

    if (!counts.empty() && counts.size() != features.rows) {
        throw DimensionMismatchError(
            "Bootstrap counts must match the number of rows"
        );
    }

    std::vector<bool> visited(features.rows, false);

    for (const auto id : object_ids) {
        if (id >= features.rows) {
            throw InvalidModelParameterError(
                "Object index out of range"
            );
        }

        if (visited[id]) {
            throw InvalidModelParameterError(
                "Object indices must be unique"
            );
        }

        visited[id] = true;

        if (!counts.empty() && counts[id] == 0) {
            throw InvalidModelParameterError(
                "Node contains an object with zero count"
            );
        }

        if (!std::isfinite(target[id])) {
            throw InvalidModelParameterError(
                "Target contains a non-finite value"
            );
        }
    }

    for (const auto feature : feature_ids) {
        if (feature >= features.cols) {
            throw InvalidModelParameterError(
                "Feature index out of range"
            );
        }

        for (const auto id : object_ids) {
            if (!std::isfinite(features(id, feature))) {
                throw InvalidModelParameterError(
                    "Features contain a non-finite value"
                );
            }
        }
    }
}

std::size_t weight_of(
    std::size_t id,
    std::span<const std::size_t> counts
) {
    return counts.empty() ? 1 : counts[id];
}

bool better_split(
    const Split& candidate,
    const std::optional<Split>& best
) {
    return !best ||
           candidate.score > best->score ||
           (candidate.score == best->score &&
            (candidate.feature < best->feature ||
             (candidate.feature == best->feature &&
              candidate.threshold < best->threshold)));
}

FeatureSplitResult search_feature(
    const NumericMatrix& features,
    std::span<const double> target,
    std::span<const std::size_t> object_ids,
    std::size_t feature,
    SplitCriterion criterion,
    std::size_t min_samples_leaf,
    std::span<const std::size_t> counts
) {
    FeatureSplitResult result;

    if (object_ids.size() < 2) {
        return result;
    }

    std::vector<std::size_t> order(
        object_ids.begin(),
        object_ids.end()
    );

    std::sort(
        order.begin(),
        order.end(),
        [&](std::size_t a, std::size_t b) {
            const double va = features(a, feature);
            const double vb = features(b, feature);

            return va < vb || (va == vb && a < b);
        }
    );

    result.is_constant =
        features(order.front(), feature) ==
        features(order.back(), feature);

    if (result.is_constant ||
        object_ids.size() / 2 < min_samples_leaf) {
        return result;
    }

    long double total_weight = 0.0L;
    long double total_sum = 0.0L;

    std::map<double, long double> class_counts;

    for (const auto id : object_ids) {
        const auto w =
            static_cast<long double>(weight_of(id, counts));

        total_weight += w;

        if (criterion == SplitCriterion::Variance) {
            total_sum +=
                w * static_cast<long double>(target[id]);
        } else {
            class_counts[target[id]] += w;
        }
    }

    long double left_weight = 0.0L;
    long double left_sum = 0.0L;

    long double left_squared_sum = 0.0L;
    long double right_squared_sum = 0.0L;

    std::map<double, long double> left_counts;

    if (criterion == SplitCriterion::Gini) {
        for (const auto& [label, count] : class_counts) {
            (void)label;
            right_squared_sum += count * count;
        }
    }

    const std::size_t n = order.size();

    for (std::size_t p = 0; p + 1 < n; ++p) {
        const auto id = order[p];

        const auto w =
            static_cast<long double>(weight_of(id, counts));

        left_weight += w;

        if (criterion == SplitCriterion::Variance) {
            left_sum +=
                w * static_cast<long double>(target[id]);
        } else {
            const double label = target[id];

            const long double a = left_counts[label];
            const long double b = class_counts.at(label) - a;

            left_squared_sum +=
                2.0L * a * w + w * w;

            right_squared_sum +=
                -2.0L * b * w + w * w;

            left_counts[label] += w;
        }

        const double current = features(id, feature);
        const double next =
            features(order[p + 1], feature);

        if (current == next ||
            p + 1 < min_samples_leaf ||
            n - (p + 1) < min_samples_leaf) {
            continue;
        }

        const long double right_weight =
            total_weight - left_weight;

        const long double right_sum =
            total_sum - left_sum;

        const long double score =
            criterion == SplitCriterion::Variance
                ? left_sum * left_sum / left_weight +
                  right_sum * right_sum / right_weight
                : left_squared_sum / left_weight +
                  right_squared_sum / right_weight;

        if (!std::isfinite(score) ||
            score > std::numeric_limits<double>::max()) {
            throw InvalidModelParameterError(
                "Split score is not finite or representable"
            );
        }

        double threshold = std::midpoint(current, next);

        if (threshold >= next) {
            threshold = current;
        }

        const Split candidate{
            feature,
            threshold,
            static_cast<double>(score)
        };

        if (better_split(candidate, result.best)) {
            result.best = candidate;
        }
    }

    return result;
}

} // namespace

std::optional<Split> SplitFinder::find_best(
    const NumericMatrix& features,
    std::span<const double> target,
    std::span<const std::size_t> object_ids,
    std::span<const std::size_t> feature_ids,
    SplitCriterion criterion,
    std::size_t min_samples_leaf,
    std::span<const std::size_t> counts
) const {
    validate_request(
        features,
        target,
        object_ids,
        feature_ids,
        min_samples_leaf,
        counts
    );

    std::optional<Split> best;

    for (const auto feature : feature_ids) {
        const auto result = search_feature(
            features,
            target,
            object_ids,
            feature,
            criterion,
            min_samples_leaf,
            counts
        );

        if (result.best &&
            better_split(*result.best, best)) {
            best = result.best;
        }
    }

    return best;
}

FeatureSplitResult SplitFinder::find_for_feature(
    const NumericMatrix& features,
    std::span<const double> target,
    std::span<const std::size_t> object_ids,
    std::size_t feature,
    SplitCriterion criterion,
    std::size_t min_samples_leaf,
    std::span<const std::size_t> counts
) const {
    const std::size_t feature_ids[] = {feature};

    validate_request(
        features,
        target,
        object_ids,
        feature_ids,
        min_samples_leaf,
        counts
    );

    return search_feature(
        features,
        target,
        object_ids,
        feature,
        criterion,
        min_samples_leaf,
        counts
    );
}
