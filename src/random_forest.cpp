#include "random_forest.hpp"
#include "ml_exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_set>
#include <utility>

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

} // namespace

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

    // Restart the forest RNG for every fit.
    std::mt19937 master(parameters_.random_state);

    std::unordered_set<std::uint32_t> used_seeds;

    std::vector<std::unique_ptr<CartRegressor>> trained;
    trained.reserve(parameters_.n_estimators);

    const std::size_t draws =
        parameters_.max_samples.value_or(features.rows);

    for (std::size_t tree_id = 0;
         tree_id < parameters_.n_estimators;
         ++tree_id) {

        const auto sample_seed =
            next_unique_seed(master, used_seeds);

        const auto feature_seed =
            next_unique_seed(master, used_seeds);

        auto tree = std::make_unique<CartRegressor>(
            parameters_.tree,
            parameters_.max_features,
            feature_seed
        );

        if (parameters_.bootstrap) {
            std::mt19937 sample_rng(sample_seed);

            std::uniform_int_distribution<std::size_t> row_dist(
                0,
                features.rows - 1
            );

            std::vector<std::size_t> counts(
                features.rows,
                0
            );

            for (std::size_t draw = 0;
                 draw < draws;
                 ++draw) {
                ++counts[row_dist(sample_rng)];
            }

            tree->fit_with_counts(
                features,
                target,
                counts
            );
        } else {
            tree->fit(features, target);
        }

        trained.push_back(std::move(tree));
    }

    // Failed refits preserve the previous trained forest.
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

    std::vector<long double> sums(
        features.rows,
        0.0L
    );

    for (const auto& tree : trees_) {
        const auto prediction = tree->predict(features);

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

    // The forest class vocabulary comes from the full target.
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

    std::mt19937 master(parameters_.random_state);

    std::unordered_set<std::uint32_t> used_seeds;

    std::vector<std::unique_ptr<CartClassifier>> trained;
    trained.reserve(parameters_.n_estimators);

    const std::size_t draws =
        parameters_.max_samples.value_or(features.rows);

    for (std::size_t tree_id = 0;
         tree_id < parameters_.n_estimators;
         ++tree_id) {

        const auto sample_seed =
            next_unique_seed(master, used_seeds);

        const auto feature_seed =
            next_unique_seed(master, used_seeds);

        auto tree = std::make_unique<CartClassifier>(
            parameters_.tree,
            parameters_.max_features,
            feature_seed
        );

        if (parameters_.bootstrap) {
            std::mt19937 sample_rng(sample_seed);

            std::uniform_int_distribution<std::size_t> row_dist(
                0,
                features.rows - 1
            );

            std::vector<std::size_t> counts(
                features.rows,
                0
            );

            for (std::size_t draw = 0;
                 draw < draws;
                 ++draw) {
                ++counts[row_dist(sample_rng)];
            }

            tree->fit_with_counts(
                features,
                target,
                counts
            );
        } else {
            tree->fit(features, target);
        }

        trained.push_back(std::move(tree));
    }

    // Replace trained state only after all trees succeed.
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

    std::vector<long double> sums(
        output.rows * output.cols,
        0.0L
    );

    for (const auto& tree : trees_) {
        const auto probabilities =
            tree->predict_proba(features);

        const auto tree_classes = tree->classes();

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
                sums[row * output.cols + global_class] +=
                    static_cast<long double>(
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










