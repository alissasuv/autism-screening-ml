#include "feature_importance.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lab5 {
namespace {

void validate_test(
    const TrainedEvaluation& trained,
    const RawDataset& test
) {
    if (test.row_count == 0 ||
        test.columns.size() != test.column_names.size() ||
        test.column_names != trained.preprocessing.column_names) {
        throw std::invalid_argument(
            "Invalid test dataset or incompatible schema"
        );
    }

    for (const auto& column : test.columns) {
        if (column.size() != test.row_count) {
            throw std::invalid_argument(
                "Inconsistent raw test column length"
            );
        }
    }
}

std::size_t find_target(
    const RawDataset& test,
    const std::string& name
) {
    const auto first = std::find(
        test.column_names.begin(),
        test.column_names.end(),
        name
    );

    if (first == test.column_names.end()) {
        throw std::invalid_argument(
            "Target column not found: " + name
        );
    }

    if (std::find(first + 1, test.column_names.end(), name)
        != test.column_names.end()) {
        throw std::invalid_argument(
            "Duplicate target column"
        );
    }

    return static_cast<std::size_t>(
        std::distance(test.column_names.begin(), first)
    );
}

double importance_drop(
    double baseline,
    double permuted,
    Scoring scoring
) {
    return higher_is_better(scoring)
        ? baseline - permuted
        : permuted - baseline;
}

void calculate_summary(FeatureImportance& feature) {
    const auto& values = feature.repeat_importances;

    const double sum = std::accumulate(
        values.begin(),
        values.end(),
        0.0
    );

    feature.mean_importance =
        sum / static_cast<double>(values.size());

    double squared_sum = 0.0;

    for (const double value : values) {
        const double difference =
            value - feature.mean_importance;

        squared_sum += difference * difference;
    }

    feature.stddev = std::sqrt(
        squared_sum / static_cast<double>(values.size())
    );
}

std::string csv_escape(const std::string& value) {
    std::string result = "\"";

    for (const char c : value) {
        if (c == '"') {
            result += "\"\"";
        } else {
            result += c;
        }
    }

    result += '"';
    return result;
}

} // namespace

PermutationImportanceResult permutation_feature_importance(
    const TrainedEvaluation& trained,
    const RawDataset& test,
    std::size_t repeats,
    std::uint32_t random_state
) {
    validate_test(trained, test);

    if (repeats == 0) {
        throw std::invalid_argument(
            "Permutation repeats must be positive"
        );
    }

    const std::size_t target_index = find_target(
        test,
        trained.preprocessing.target_name
    );

    PermutationImportanceResult result;

    // Calculate baseline exactly once, before permutations.
    result.baseline_score = score_final_model(
        trained,
        test
    );

    if (!std::isfinite(result.baseline_score)) {
        throw std::runtime_error(
            "Baseline score must be finite"
        );
    }

    // One RNG for the entire experiment.
    // Its state advances across features and repetitions.
    std::mt19937 rng(random_state);

    for (std::size_t column = 0;
         column < test.columns.size();
         ++column) {

        // Never permute the target.
        if (column == target_index) {
            continue;
        }

        FeatureImportance feature;
        feature.feature_index = column;
        feature.feature_name = test.column_names[column];
        feature.repeat_importances.reserve(repeats);

        const auto& original_values = test.columns[column];

        for (std::size_t repeat = 0;
             repeat < repeats;
             ++repeat) {

            // Each repetition starts from the ORIGINAL test.
            RawDataset permuted = test;

            std::vector<std::size_t> order(test.row_count);
            std::iota(order.begin(), order.end(), 0);

            std::shuffle(
                order.begin(),
                order.end(),
                rng
            );

            // Move complete raw cell values.
            // std::optional also preserves missing values.
            for (std::size_t row = 0;
                 row < test.row_count;
                 ++row) {
                permuted.columns[column][row] =
                    original_values[order[row]];
            }

            // Reuse the original fitted preprocessing and
            // already trained Random Forest.
            const double permuted_score = score_final_model(
                trained,
                permuted
            );

            if (!std::isfinite(permuted_score)) {
                throw std::runtime_error(
                    "Permuted score must be finite"
                );
            }

            feature.repeat_importances.push_back(
                importance_drop(
                    result.baseline_score,
                    permuted_score,
                    trained.scoring
                )
            );
        }

        calculate_summary(feature);

        if (!std::isfinite(feature.mean_importance) ||
            !std::isfinite(feature.stddev)) {
            throw std::runtime_error(
                "Non-finite feature importance"
            );
        }

        result.features.push_back(std::move(feature));
    }

    // The largest decrease in quality comes first.
    // Ties retain the original column order.
    std::stable_sort(
        result.features.begin(),
        result.features.end(),
        [](const FeatureImportance& left,
           const FeatureImportance& right) {
            return left.mean_importance >
                   right.mean_importance;
        }
    );

    return result;
}

void save_feature_importance(
    const PermutationImportanceResult& result,
    const std::string& filename
) {
    std::ofstream out(filename);

    if (!out) {
        throw std::runtime_error(
            "Cannot create importance CSV: " + filename
        );
    }

    out << "feature_index,feature,mean_importance,stddev\n";
    out << std::setprecision(17);

    for (const auto& feature : result.features) {
        out << feature.feature_index << ','
            << csv_escape(feature.feature_name) << ','
            << feature.mean_importance << ','
            << feature.stddev << '\n';
    }

    if (!out) {
        throw std::runtime_error(
            "Failed to write feature importance CSV"
        );
    }
}

} // namespace lab5

