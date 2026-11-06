#include "split_finder.hpp"
#include "ml_exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

std::optional<Split> SplitFinder::find_best(
    const NumericMatrix& features,
    std::span<const double> target,
    std::span<const std::size_t> object_ids,
    std::span<const std::size_t> feature_ids,
    SplitCriterion criterion,
    std::size_t min_samples_leaf
) const {
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
        features.rows > std::numeric_limits<std::size_t>::max()
                            / features.cols) {
        throw DimensionMismatchError(
            "Feature matrix dimensions overflow"
        );
    }

    if (features.values.size() != features.rows * features.cols) {
        throw DimensionMismatchError(
            "Invalid feature matrix dimensions"
        );
    }

    if (object_ids.size() < 2 ||
        object_ids.size() / 2 < min_samples_leaf ||
        feature_ids.empty()) {
        return std::nullopt;
    }

    for (const std::size_t id : object_ids) {
        if (id >= features.rows) {
            throw InvalidModelParameterError(
                "Object index out of range"
            );
        }

        if (!std::isfinite(target[id])) {
            throw InvalidModelParameterError(
                "Target contains a non-finite value"
            );
        }
    }

    for (const std::size_t feature : feature_ids) {
        if (feature >= features.cols) {
            throw InvalidModelParameterError(
                "Feature index out of range"
            );
        }

        for (const std::size_t id : object_ids) {
            if (!std::isfinite(features(id, feature))) {
                throw InvalidModelParameterError(
                    "Features contain a non-finite value"
                );
            }
        }
    }

    const std::size_t n = object_ids.size();

    long double total_sum = 0.0L;
    std::map<double, std::size_t> class_counts;

    for (const std::size_t id : object_ids) {
        if (criterion == SplitCriterion::Variance) {
            total_sum += static_cast<long double>(target[id]);
        } else {
            ++class_counts[target[id]];
        }
    }

    std::map<double, std::size_t> class_ids;
    std::size_t next_class_id = 0;

    for (const auto& [label, count] : class_counts) {
        (void)count;
        class_ids[label] = next_class_id++;
    }

    std::vector<std::size_t> encoded_target(
        features.rows, 0
    );
    std::vector<std::size_t> total_counts(
        class_ids.size(), 0
    );

    if (criterion == SplitCriterion::Gini) {
        for (const std::size_t id : object_ids) {
            const auto class_id = class_ids.at(target[id]);
            encoded_target[id] = class_id;
            ++total_counts[class_id];
        }
    }

    std::optional<Split> best;

    for (const std::size_t feature : feature_ids) {
        std::vector<std::size_t> order(
            object_ids.begin(), object_ids.end()
        );

        std::sort(
            order.begin(),
            order.end(),
            [&](std::size_t a, std::size_t b) {
                const double va = features(a, feature);
                const double vb = features(b, feature);

                if (va != vb) {
                    return va < vb;
                }

                return a < b;
            }
        );

        long double left_sum = 0.0L;
        std::size_t left_size = 0;

        std::vector<std::size_t> left_counts(
            total_counts.size(), 0
        );
        std::vector<std::size_t> right_counts =
            total_counts;

        long double left_squared_sum = 0.0L;
        long double right_squared_sum = 0.0L;

        if (criterion == SplitCriterion::Gini) {
            for (const auto count : right_counts) {
                right_squared_sum +=
                    static_cast<long double>(count) * count;
            }
        }

        for (std::size_t p = 0; p + 1 < n; ++p) {
            const std::size_t id = order[p];

            ++left_size;
            const std::size_t right_size = n - left_size;

            if (criterion == SplitCriterion::Variance) {
                left_sum +=
                    static_cast<long double>(target[id]);
            } else {
                const auto class_id = encoded_target[id];

                const long double a =
                    static_cast<long double>(
                        left_counts[class_id]
                    );
                const long double b =
                    static_cast<long double>(
                        right_counts[class_id]
                    );

                left_squared_sum += 2.0L * a + 1.0L;
                right_squared_sum += 1.0L - 2.0L * b;

                ++left_counts[class_id];
                --right_counts[class_id];
            }

            const double current =
                features(id, feature);
            const double next =
                features(order[p + 1], feature);

            if (current == next) {
                continue;
            }

            if (left_size < min_samples_leaf ||
                right_size < min_samples_leaf) {
                continue;
            }

            long double score = 0.0L;

            if (criterion == SplitCriterion::Variance) {
                const long double right_sum =
                    total_sum - left_sum;

                score =
                    left_sum * left_sum / left_size +
                    right_sum * right_sum / right_size;
            } else {
                score =
                    left_squared_sum / left_size +
                    right_squared_sum / right_size;
            }

            if (!std::isfinite(score) ||
                score > std::numeric_limits<double>::max()) {
                throw InvalidModelParameterError(
                    "Split score is not finite or representable"
                );
            }

            double threshold =
                std::midpoint(current, next);

            if (threshold >= next) {
                threshold = current;
            }

            const double candidate_score =
                static_cast<double>(score);

            const bool is_better =
                !best ||
                candidate_score > best->score ||
                (candidate_score == best->score &&
                    (feature < best->feature ||
                     (feature == best->feature &&
                      threshold < best->threshold)));

            if (is_better) {
                best = Split{
                    feature,
                    threshold,
                    candidate_score
                };
            }
        }
    }

    return best;
}