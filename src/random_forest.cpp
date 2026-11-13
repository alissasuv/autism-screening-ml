#include "random_forest.hpp"
#include "forest_parallel.hpp"
#include "ml_exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

void validate_parameters(const RandomForestParameters& p) {
    if (p.n_estimators == 0) {
        throw InvalidModelParameterError(
            "n_estimators must be at least 1"
        );
    }

    if (p.n_jobs == 0) {
        throw InvalidModelParameterError(
            "n_jobs must be at least 1"
        );
    }

    if (p.tree.min_samples_split < 2 ||
        p.tree.min_samples_leaf == 0) {
        throw InvalidModelParameterError(
            "Invalid CART sample limits"
        );
    }

    if (p.max_features.mode == FeatureMode::Count &&
        p.max_features.count == 0) {
        throw InvalidModelParameterError(
            "max_features count must be positive"
        );
    }

    if (!p.bootstrap && p.max_samples.has_value()) {
        throw InvalidModelParameterError(
            "max_samples requires bootstrap=true"
        );
    }

    if (p.max_samples && *p.max_samples == 0) {
        throw InvalidModelParameterError(
            "max_samples must be positive"
        );
    }
}

void validate_training(
    const NumericMatrix& x,
    std::span<const double> y,
    const RandomForestParameters& p
) {
    if (x.rows == 0 || x.cols == 0) {
        throw InvalidModelParameterError(
            "Training matrix must not be empty"
        );
    }

    if (x.rows != y.size() ||
        x.rows >
            std::numeric_limits<std::size_t>::max() / x.cols ||
        x.values.size() != x.rows * x.cols) {
        throw DimensionMismatchError(
            "Invalid training dimensions"
        );
    }

    if (p.max_samples && *p.max_samples > x.rows) {
        throw InvalidModelParameterError(
            "max_samples exceeds number of rows"
        );
    }

    if (p.max_features.mode == FeatureMode::Count &&
        p.max_features.count > x.cols) {
        throw InvalidModelParameterError(
            "max_features exceeds number of columns"
        );
    }

    for (const auto value : x.values) {
        if (!std::isfinite(value)) {
            throw InvalidModelParameterError(
                "Non-finite training feature"
            );
        }
    }

    for (const auto value : y) {
        if (!std::isfinite(value)) {
            throw InvalidModelParameterError(
                "Non-finite target"
            );
        }
    }
}

std::uint32_t next_unique_seed(
    std::mt19937& generator,
    std::unordered_set<std::uint32_t>& used
) {
    std::uint32_t candidate;

    do {
        candidate = generator();
    } while (!used.insert(candidate).second);

    return candidate;
}

struct TreeSeeds {
    std::uint32_t sample;
    std::uint32_t feature;
};

std::vector<TreeSeeds> make_tree_seeds(
    const RandomForestParameters& parameters
) {
    std::mt19937 master(parameters.random_state);
    std::unordered_set<std::uint32_t> used;

    std::vector<TreeSeeds> seeds;
    seeds.reserve(parameters.n_estimators);

    for (std::size_t i = 0;
         i < parameters.n_estimators;
         ++i) {
        seeds.push_back({
            next_unique_seed(master, used),
            next_unique_seed(master, used)
        });
    }

    return seeds;
}

std::vector<std::size_t> bootstrap_counts(
    std::size_t rows,
    std::size_t draws,
    std::uint32_t seed
) {
    std::mt19937 rng(seed);

    std::uniform_int_distribution<std::size_t> distribution(
        0,
        rows - 1
    );

    std::vector<std::size_t> counts(rows, 0);

    for (std::size_t i = 0; i < draws; ++i) {
        ++counts[distribution(rng)];
    }

    return counts;
}

} // namespace

// ============================================================
// Random Forest Regressor
// ============================================================

RandomForestRegressor::RandomForestRegressor(
    RandomForestParameters parameters
)
    : parameters_(parameters) {
    validate_parameters(parameters_);
}

void RandomForestRegressor::fit(
    const NumericMatrix& features,
    std::span<const double> target
) {
    validate_training(features, target, parameters_);

    const auto seeds = make_tree_seeds(parameters_);

    const std::size_t draws =
        parameters_.max_samples.value_or(features.rows);

    // Each worker writes only to its own indexed slot.
    std::vector<std::unique_ptr<CartRegressor>> trained(
        parameters_.n_estimators
    );

    forest_parallel::for_each_index(
        parameters_.n_estimators,
        parameters_.n_jobs,
        [&](std::size_t i) {
            auto tree = std::make_unique<CartRegressor>(
                parameters_.tree,
                parameters_.max_features,
                seeds[i].feature
            );

            if (parameters_.bootstrap) {
                const auto counts = bootstrap_counts(
                    features.rows,
                    draws,
                    seeds[i].sample
                );

                tree->fit_with_counts(
                    features,
                    target,
                    counts
                );
            } else {
                tree->fit(features, target);
            }

            trained[i] = std::move(tree);
        }
    );

    // The previous forest survives a failed refit.
    trees_.swap(trained);
}

std::vector<double> RandomForestRegressor::predict(
    const NumericMatrix& features
) const {
    if (trees_.empty()) {
        throw InvalidModelStateError(
            "Random Forest has not been fitted"
        );
    }

    // Parallel tasks write separate prediction vectors.
    std::vector<std::vector<double>> tree_predictions(
        trees_.size()
    );

    forest_parallel::for_each_index(
        trees_.size(),
        parameters_.n_jobs,
        [&](std::size_t i) {
            tree_predictions[i] =
                trees_[i]->predict(features);
        }
    );

    // Accumulate strictly in tree-index order.
    std::vector<long double> sums(
        features.rows,
        0.0L
    );

    for (const auto& prediction : tree_predictions) {
        for (std::size_t row = 0;
             row < prediction.size();
             ++row) {
            sums[row] +=
                static_cast<long double>(prediction[row]);
        }
    }

    std::vector<double> result(features.rows);

    for (std::size_t row = 0;
         row < features.rows;
         ++row) {
        const long double average =
            sums[row] / trees_.size();

        if (!std::isfinite(average) ||
            std::abs(average) >
                std::numeric_limits<double>::max()) {
            throw InvalidModelStateError(
                "Forest prediction is not finite"
            );
        }

        result[row] = static_cast<double>(average);
    }

    return result;
}

std::size_t RandomForestRegressor::tree_count() const noexcept {
    return trees_.size();
}

std::vector<std::size_t>
RandomForestRegressor::tree_node_counts() const {
    if (trees_.empty()) {
        throw InvalidModelStateError(
            "Random Forest has not been fitted"
        );
    }

    std::vector<std::size_t> result;
    result.reserve(trees_.size());

    for (const auto& tree : trees_) {
        result.push_back(tree->node_count());
    }

    return result;
}

std::vector<std::size_t>
RandomForestRegressor::tree_depths() const {
    if (trees_.empty()) {
        throw InvalidModelStateError(
            "Random Forest has not been fitted"
        );
    }

    std::vector<std::size_t> result;
    result.reserve(trees_.size());

    for (const auto& tree : trees_) {
        result.push_back(tree->depth());
    }

    return result;
}

// ============================================================
// Random Forest Classifier
// ============================================================

RandomForestParameters classifier_default_parameters() {
    RandomForestParameters parameters;
    parameters.max_features = {FeatureMode::Sqrt, 1};
    return parameters;
}

RandomForestClassifier::RandomForestClassifier(
    RandomForestParameters parameters
)
    : parameters_(parameters) {
    validate_parameters(parameters_);
}

void RandomForestClassifier::fit(
    const NumericMatrix& features,
    std::span<const double> target
) {
    validate_training(features, target, parameters_);

    // Classes are determined from the full training set.
    std::vector<double> new_classes(
        target.begin(),
        target.end()
    );

    std::sort(new_classes.begin(), new_classes.end());

    new_classes.erase(
        std::unique(
            new_classes.begin(),
            new_classes.end()
        ),
        new_classes.end()
    );

    const auto seeds = make_tree_seeds(parameters_);

    const std::size_t draws =
        parameters_.max_samples.value_or(features.rows);

    std::vector<std::unique_ptr<CartClassifier>> trained(
        parameters_.n_estimators
    );

    forest_parallel::for_each_index(
        parameters_.n_estimators,
        parameters_.n_jobs,
        [&](std::size_t i) {
            auto tree = std::make_unique<CartClassifier>(
                parameters_.tree,
                parameters_.max_features,
                seeds[i].feature
            );

            if (parameters_.bootstrap) {
                const auto counts = bootstrap_counts(
                    features.rows,
                    draws,
                    seeds[i].sample
                );

                tree->fit_with_counts(
                    features,
                    target,
                    counts
                );
            } else {
                tree->fit(features, target);
            }

            trained[i] = std::move(tree);
        }
    );

    // Commit the newly trained state only after success.
    trees_.swap(trained);
    classes_.swap(new_classes);
}

NumericMatrix RandomForestClassifier::predict_proba(
    const NumericMatrix& features
) const {
    if (trees_.empty()) {
        throw InvalidModelStateError(
            "Random Forest has not been fitted"
        );
    }

    NumericMatrix output;
    output.rows = features.rows;
    output.cols = classes_.size();
    output.layout = MatrixLayout::RowMajor;

    if (output.cols != 0 &&
        output.rows >
            std::numeric_limits<std::size_t>::max() /
                output.cols) {
        throw DimensionMismatchError(
            "Probability matrix size overflow"
        );
    }

    // Compute each tree's probabilities independently.
    std::vector<NumericMatrix> tree_probabilities(
        trees_.size()
    );

    forest_parallel::for_each_index(
        trees_.size(),
        parameters_.n_jobs,
        [&](std::size_t i) {
            tree_probabilities[i] =
                trees_[i]->predict_proba(features);
        }
    );

    // Combine results in deterministic tree-index order.
    std::vector<long double> sums(
        output.rows * output.cols,
        0.0L
    );

    for (std::size_t tree_id = 0;
         tree_id < trees_.size();
         ++tree_id) {
        const auto& probabilities =
            tree_probabilities[tree_id];

        const auto tree_classes =
            trees_[tree_id]->classes();

        for (std::size_t local_class = 0;
             local_class < tree_classes.size();
             ++local_class) {
            const auto it = std::lower_bound(
                classes_.begin(),
                classes_.end(),
                tree_classes[local_class]
            );

            if (it == classes_.end() ||
                *it != tree_classes[local_class]) {
                throw InvalidModelStateError(
                    "Tree class not found in forest classes"
                );
            }

            const auto global_class =
                static_cast<std::size_t>(
                    it - classes_.begin()
                );

            for (std::size_t row = 0;
                 row < features.rows;
                 ++row) {
                sums[
                    row * output.cols + global_class
                ] += static_cast<long double>(
                    probabilities(row, local_class)
                );
            }
        }
    }

    output.values.reserve(sums.size());

    for (const auto sum : sums) {
        const long double average =
            sum / trees_.size();

        if (!std::isfinite(average) ||
            average < 0.0L ||
            average > 1.0L) {
            throw InvalidModelStateError(
                "Invalid forest class probability"
            );
        }

        output.values.push_back(
            static_cast<double>(average)
        );
    }

    return output;
}

std::vector<double> RandomForestClassifier::predict(
    const NumericMatrix& features
) const {
    const auto probabilities = predict_proba(features);

    std::vector<double> result;
    result.reserve(features.rows);

    for (std::size_t row = 0;
         row < features.rows;
         ++row) {
        std::size_t winner = 0;

        for (std::size_t col = 1;
             col < classes_.size();
             ++col) {
            if (probabilities(row, col) >
                probabilities(row, winner)) {
                winner = col;
            }
        }

        result.push_back(classes_[winner]);
    }

    return result;
}

std::span<const double> RandomForestClassifier::classes() const {
    if (trees_.empty()) {
        throw InvalidModelStateError(
            "Random Forest has not been fitted"
        );
    }

    return classes_;
}

std::size_t RandomForestClassifier::tree_count() const noexcept {
    return trees_.size();
}

std::vector<std::size_t>
RandomForestClassifier::tree_node_counts() const {
    if (trees_.empty()) {
        throw InvalidModelStateError(
            "Random Forest has not been fitted"
        );
    }

    std::vector<std::size_t> result;
    result.reserve(trees_.size());

    for (const auto& tree : trees_) {
        result.push_back(tree->node_count());
    }

    return result;
}

std::vector<std::size_t>
RandomForestClassifier::tree_depths() const {
    if (trees_.empty()) {
        throw InvalidModelStateError(
            "Random Forest has not been fitted"
        );
    }

    std::vector<std::size_t> result;
    result.reserve(trees_.size());

    for (const auto& tree : trees_) {
        result.push_back(tree->depth());
    }

    return result;
}




