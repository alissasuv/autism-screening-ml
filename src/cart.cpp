#include "cart.hpp"
#include "ml_exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

struct CartTree::Node {
    bool is_leaf = true;
    std::size_t feature = 0;
    double threshold = 0.0;
    double prediction = 0.0;

    std::vector<double> probabilities;

    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
};

CartTree::CartTree(
    SplitCriterion criterion,
    CartParameters parameters,
    FeatureSearchParams feature_search,
    std::uint32_t random_state
)
    : criterion_(criterion),
      parameters_(parameters),
      feature_search_(feature_search),
      random_state_(random_state) {
    if (parameters_.min_samples_split < 2) {
        throw InvalidModelParameterError(
            "min_samples_split must be at least 2"
        );
    }

    if (parameters_.min_samples_leaf == 0) {
        throw InvalidModelParameterError(
            "min_samples_leaf must be positive"
        );
    }

    if (feature_search_.mode == FeatureMode::Count &&
        feature_search_.count == 0) {
        throw InvalidModelParameterError(
            "Feature count must be positive"
        );
    }
}

CartTree::~CartTree() = default;
CartTree::CartTree(CartTree&&) noexcept = default;
CartTree& CartTree::operator=(CartTree&&) noexcept = default;

std::unique_ptr<CartTree::Node> CartTree::build(
    const NumericMatrix& features,
    std::span<const double> target,
    std::span<const std::size_t> object_ids,
    std::span<const std::size_t> counts,
    std::size_t current_depth,
    std::span<const double> classes,
    std::mt19937& rng,
    std::size_t& nodes,
    std::size_t& leaves,
    std::size_t& deepest
) const {
    auto node = std::make_unique<Node>();

    ++nodes;
    deepest = std::max(deepest, current_depth);

    const auto weight_of = [&](std::size_t id) -> std::size_t {
        return counts.empty() ? 1 : counts[id];
    };

    long double total_weight = 0.0L;

    for (const auto id : object_ids) {
        total_weight +=
            static_cast<long double>(weight_of(id));
    }

    if (criterion_ == SplitCriterion::Variance) {
        long double sum = 0.0L;

        for (const auto id : object_ids) {
            sum +=
                static_cast<long double>(weight_of(id)) *
                target[id];
        }

        const long double average = sum / total_weight;

        if (!std::isfinite(average) ||
            std::abs(average) >
                std::numeric_limits<double>::max()) {
            throw InvalidModelParameterError(
                "Regression leaf prediction is not finite"
            );
        }

        node->prediction = static_cast<double>(average);
    } else {
        std::vector<long double> class_weights(
            classes.size(),
            0.0L
        );

        for (const auto id : object_ids) {
            const auto it = std::lower_bound(
                classes.begin(),
                classes.end(),
                target[id]
            );

            const auto index = static_cast<std::size_t>(
                it - classes.begin()
            );

            class_weights[index] +=
                static_cast<long double>(weight_of(id));
        }

        std::size_t winner = 0;

        node->probabilities.reserve(classes.size());

        for (std::size_t i = 0;
             i < class_weights.size();
             ++i) {
            node->probabilities.push_back(
                static_cast<double>(
                    class_weights[i] / total_weight
                )
            );

            if (class_weights[i] > class_weights[winner]) {
                winner = i;
            }
        }

        node->prediction = classes[winner];
    }

    bool all_equal = true;

    for (std::size_t i = 1;
         i < object_ids.size();
         ++i) {
        if (target[object_ids[i]] != target[object_ids[0]]) {
            all_equal = false;
            break;
        }
    }

    if ((parameters_.max_depth &&
         current_depth >= *parameters_.max_depth) ||
        object_ids.size() < parameters_.min_samples_split ||
        object_ids.size() / 2 <
            parameters_.min_samples_leaf ||
        all_equal) {
        ++leaves;
        return node;
    }

    std::vector<std::size_t> feature_order(features.cols);

    std::iota(
        feature_order.begin(),
        feature_order.end(),
        0
    );

    std::size_t budget = features.cols;

    if (feature_search_.mode == FeatureMode::Sqrt) {
        budget = std::max<std::size_t>(
            1,
            static_cast<std::size_t>(
                std::sqrt(
                    static_cast<long double>(features.cols)
                )
            )
        );
    } else if (feature_search_.mode == FeatureMode::Count) {
        budget = feature_search_.count;
    }

    // Preserve Lab 3 ordering for the All mode.
    if (feature_search_.mode != FeatureMode::All) {
        std::shuffle(
            feature_order.begin(),
            feature_order.end(),
            rng
        );
    }

    std::optional<Split> best;

    std::size_t checked = 0;
    bool found_nonconstant = false;

    for (const auto feature : feature_order) {
        if (checked >= budget && found_nonconstant) {
            break;
        }

        const auto result = split_finder_.find_for_feature(
            features,
            target,
            object_ids,
            feature,
            criterion_,
            parameters_.min_samples_leaf,
            counts
        );

        ++checked;

        if (!result.is_constant) {
            found_nonconstant = true;
        }

        if (result.best &&
            (!best ||
             result.best->score > best->score ||
             (result.best->score == best->score &&
              (result.best->feature < best->feature ||
               (result.best->feature == best->feature &&
                result.best->threshold <
                    best->threshold))))) {
            best = result.best;
        }
    }

    if (!best) {
        ++leaves;
        return node;
    }

    std::vector<std::size_t> left;
    std::vector<std::size_t> right;

    left.reserve(object_ids.size());
    right.reserve(object_ids.size());

    for (const auto id : object_ids) {
        if (features(id, best->feature) <= best->threshold) {
            left.push_back(id);
        } else {
            right.push_back(id);
        }
    }

    if (left.size() < parameters_.min_samples_leaf ||
        right.size() < parameters_.min_samples_leaf) {
        throw InvalidModelStateError(
            "SplitFinder returned an invalid partition"
        );
    }

    node->feature = best->feature;
    node->threshold = best->threshold;

    node->left = build(
        features,
        target,
        left,
        counts,
        current_depth + 1,
        classes,
        rng,
        nodes,
        leaves,
        deepest
    );

    node->right = build(
        features,
        target,
        right,
        counts,
        current_depth + 1,
        classes,
        rng,
        nodes,
        leaves,
        deepest
    );

    node->is_leaf = false;

    return node;
}

void CartTree::fit(
    const NumericMatrix& features,
    std::span<const double> target
) {
    fit_with_counts(features, target, {});
}

void CartTree::fit_with_counts(
    const NumericMatrix& features,
    std::span<const double> target,
    std::span<const std::size_t> counts
) {
    if (features.rows == 0 || features.cols == 0) {
        throw InvalidModelParameterError(
            "Training matrix must be non-empty"
        );
    }

    if (features.rows != target.size()) {
        throw DimensionMismatchError(
            "Features and target have different sizes"
        );
    }

    if (features.rows >
            std::numeric_limits<std::size_t>::max() /
                features.cols ||
        features.values.size() !=
            features.rows * features.cols) {
        throw DimensionMismatchError(
            "Invalid training matrix dimensions"
        );
    }

    if (!counts.empty() && counts.size() != features.rows) {
        throw DimensionMismatchError(
            "Bootstrap counts must match training rows"
        );
    }

    if (feature_search_.mode == FeatureMode::Count &&
        feature_search_.count > features.cols) {
        throw InvalidModelParameterError(
            "Requested feature count exceeds columns"
        );
    }

    for (const double value : features.values) {
        if (!std::isfinite(value)) {
            throw InvalidModelParameterError(
                "Non-finite feature value"
            );
        }
    }

    for (const double value : target) {
        if (!std::isfinite(value)) {
            throw InvalidModelParameterError(
                "Non-finite target value"
            );
        }
    }

    std::vector<std::size_t> object_ids;
    object_ids.reserve(features.rows);

    for (std::size_t id = 0; id < features.rows; ++id) {
        if (counts.empty() || counts[id] != 0) {
            object_ids.push_back(id);
        }
    }

    if (object_ids.empty()) {
        throw InvalidModelParameterError(
            "At least one bootstrap count must be positive"
        );
    }

    std::vector<double> new_classes;

    if (criterion_ == SplitCriterion::Gini) {
        new_classes.reserve(object_ids.size());

        for (const auto id : object_ids) {
            new_classes.push_back(target[id]);
        }

        std::sort(
            new_classes.begin(),
            new_classes.end()
        );

        new_classes.erase(
            std::unique(
                new_classes.begin(),
                new_classes.end()
            ),
            new_classes.end()
        );
    }

    // Restart random generation for every fit.
    std::mt19937 rng(random_state_);

    std::size_t nodes = 0;
    std::size_t leaves = 0;
    std::size_t deepest = 0;

    auto new_root = build(
        features,
        target,
        object_ids,
        counts,
        0,
        new_classes,
        rng,
        nodes,
        leaves,
        deepest
    );

    // Replace the old trained state only after success.
    root_ = std::move(new_root);
    classes_ = std::move(new_classes);

    feature_count_ = features.cols;
    node_count_ = nodes;
    leaf_count_ = leaves;
    max_depth_ = deepest;
}

void CartTree::validate_prediction(
    const NumericMatrix& features
) const {
    if (!root_) {
        throw InvalidModelStateError(
            "Model has not been fitted"
        );
    }

    if (features.cols != feature_count_) {
        throw DimensionMismatchError(
            "Prediction feature count differs from training"
        );
    }

    if (features.cols > 0 &&
        features.rows >
            std::numeric_limits<std::size_t>::max() /
                features.cols) {
        throw DimensionMismatchError(
            "Prediction matrix dimensions overflow"
        );
    }

    if (features.values.size() !=
        features.rows * features.cols) {
        throw DimensionMismatchError(
            "Invalid prediction matrix dimensions"
        );
    }

    for (const double value : features.values) {
        if (!std::isfinite(value)) {
            throw InvalidModelParameterError(
                "Non-finite prediction feature value"
            );
        }
    }
}

const CartTree::Node& CartTree::find_leaf(
    const NumericMatrix& features,
    std::size_t row
) const {
    const Node* node = root_.get();

    while (!node->is_leaf) {
        node =
            features(row, node->feature) <= node->threshold
                ? node->left.get()
                : node->right.get();
    }

    return *node;
}

std::vector<double> CartTree::predict(
    const NumericMatrix& features
) const {
    validate_prediction(features);

    std::vector<double> result;
    result.reserve(features.rows);

    for (std::size_t row = 0;
         row < features.rows;
         ++row) {
        result.push_back(
            find_leaf(features, row).prediction
        );
    }

    return result;
}

NumericMatrix CartTree::predict_proba(
    const NumericMatrix& features
) const {
    if (criterion_ != SplitCriterion::Gini) {
        throw InvalidModelStateError(
            "Probabilities are only available for classification"
        );
    }

    validate_prediction(features);

    NumericMatrix output;

    output.rows = features.rows;
    output.cols = classes_.size();
    output.layout = MatrixLayout::RowMajor;

    output.values.reserve(
        output.rows * output.cols
    );

    for (std::size_t row = 0;
         row < features.rows;
         ++row) {
        const auto& probabilities =
            find_leaf(features, row).probabilities;

        output.values.insert(
            output.values.end(),
            probabilities.begin(),
            probabilities.end()
        );
    }

    return output;
}

std::span<const double> CartTree::classes() const {
    if (!root_) {
        throw InvalidModelStateError(
            "Model has not been fitted"
        );
    }

    if (criterion_ != SplitCriterion::Gini) {
        throw InvalidModelStateError(
            "Regression does not have classes"
        );
    }

    return classes_;
}

std::size_t CartTree::node_count() const noexcept {
    return node_count_;
}

std::size_t CartTree::leaf_count() const noexcept {
    return leaf_count_;
}

std::size_t CartTree::depth() const noexcept {
    return max_depth_;
}

std::vector<CartSplitInfo> CartTree::splits(
    std::size_t limit
) const {
    if (!root_) {
        throw InvalidModelStateError(
            "Model has not been fitted"
        );
    }

    std::vector<CartSplitInfo> result;

    if (limit == 0) {
        return result;
    }

    std::vector<std::pair<const Node*, std::size_t>> stack;
    stack.emplace_back(root_.get(), 0);

    while (!stack.empty() && result.size() < limit) {
        const auto [node, level] = stack.back();
        stack.pop_back();

        if (node->is_leaf) {
            continue;
        }

        result.push_back({
            level,
            node->feature,
            node->threshold
        });

        stack.emplace_back(
            node->right.get(),
            level + 1
        );

        stack.emplace_back(
            node->left.get(),
            level + 1
        );
    }

    return result;
}

// CART regression

CartRegressor::CartRegressor(
    CartParameters parameters,
    FeatureSearchParams feature_search,
    std::uint32_t random_state
)
    : tree_(
        SplitCriterion::Variance,
        parameters,
        feature_search,
        random_state
    ) {}

void CartRegressor::fit(
    const NumericMatrix& features,
    std::span<const double> target
) {
    tree_.fit(features, target);
}

void CartRegressor::fit_with_counts(
    const NumericMatrix& features,
    std::span<const double> target,
    std::span<const std::size_t> counts
) {
    tree_.fit_with_counts(features, target, counts);
}

std::vector<double> CartRegressor::predict(
    const NumericMatrix& features
) const {
    return tree_.predict(features);
}

std::size_t CartRegressor::node_count() const noexcept {
    return tree_.node_count();
}

std::size_t CartRegressor::leaf_count() const noexcept {
    return tree_.leaf_count();
}

std::size_t CartRegressor::depth() const noexcept {
    return tree_.depth();
}

std::vector<CartSplitInfo> CartRegressor::splits(
    std::size_t limit
) const {
    return tree_.splits(limit);
}

// CART classification

CartClassifier::CartClassifier(
    CartParameters parameters,
    FeatureSearchParams feature_search,
    std::uint32_t random_state
)
    : tree_(
        SplitCriterion::Gini,
        parameters,
        feature_search,
        random_state
    ) {}

void CartClassifier::fit(
    const NumericMatrix& features,
    std::span<const double> target
) {
    tree_.fit(features, target);
}

void CartClassifier::fit_with_counts(
    const NumericMatrix& features,
    std::span<const double> target,
    std::span<const std::size_t> counts
) {
    tree_.fit_with_counts(features, target, counts);
}

std::vector<double> CartClassifier::predict(
    const NumericMatrix& features
) const {
    return tree_.predict(features);
}

NumericMatrix CartClassifier::predict_proba(
    const NumericMatrix& features
) const {
    return tree_.predict_proba(features);
}

std::span<const double> CartClassifier::classes() const {
    return tree_.classes();
}

std::size_t CartClassifier::node_count() const noexcept {
    return tree_.node_count();
}

std::size_t CartClassifier::leaf_count() const noexcept {
    return tree_.leaf_count();
}

std::size_t CartClassifier::depth() const noexcept {
    return tree_.depth();
}

std::vector<CartSplitInfo> CartClassifier::splits(
    std::size_t limit
) const {
    return tree_.splits(limit);
}

