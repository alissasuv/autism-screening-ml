#pragma once

#include "model.hpp"
#include "split_finder.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <vector>

struct CartParameters {
    // nullopt means unlimited depth.
    std::optional<std::size_t> max_depth = 5;
    std::size_t min_samples_split = 2;
    std::size_t min_samples_leaf = 1;
};

enum class FeatureMode {
    All,
    Sqrt,
    Count
};

struct FeatureSearchParams {
    FeatureMode mode = FeatureMode::All;
    std::size_t count = 1;
};

struct CartSplitInfo {
    std::size_t depth = 0;
    std::size_t feature = 0;
    double threshold = 0.0;
};

class CartTree {
public:
    CartTree(
        SplitCriterion criterion,
        CartParameters parameters,
        FeatureSearchParams feature_search = {},
        std::uint32_t random_state = 0
    );

    ~CartTree();

    CartTree(const CartTree&) = delete;
    CartTree& operator=(const CartTree&) = delete;

    CartTree(CartTree&&) noexcept;
    CartTree& operator=(CartTree&&) noexcept;

    void fit(
        const NumericMatrix& features,
        std::span<const double> target
    );

    void fit_with_counts(
        const NumericMatrix& features,
        std::span<const double> target,
        std::span<const std::size_t> counts
    );

    std::vector<double> predict(
        const NumericMatrix& features
    ) const;

    NumericMatrix predict_proba(
        const NumericMatrix& features
    ) const;

    std::span<const double> classes() const;

    std::size_t node_count() const noexcept;
    std::size_t leaf_count() const noexcept;
    std::size_t depth() const noexcept;

    std::vector<CartSplitInfo> splits(
        std::size_t limit = 5
    ) const;

private:
    struct Node;

    SplitCriterion criterion_;
    CartParameters parameters_;
    FeatureSearchParams feature_search_;
    std::uint32_t random_state_;

    SplitFinder split_finder_;

    std::unique_ptr<Node> root_;
    std::vector<double> classes_;

    std::size_t feature_count_ = 0;
    std::size_t node_count_ = 0;
    std::size_t leaf_count_ = 0;
    std::size_t max_depth_ = 0;

    std::unique_ptr<Node> build(
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
    ) const;

    void validate_prediction(
        const NumericMatrix& features
    ) const;

    const Node& find_leaf(
        const NumericMatrix& features,
        std::size_t row
    ) const;
};

class CartRegressor final : public IRegressor {
public:
    explicit CartRegressor(
        CartParameters parameters = {},
        FeatureSearchParams feature_search = {},
        std::uint32_t random_state = 0
    );

    void fit(
        const NumericMatrix& features,
        std::span<const double> target
    ) override;

    void fit_with_counts(
        const NumericMatrix& features,
        std::span<const double> target,
        std::span<const std::size_t> counts
    );

    std::vector<double> predict(
        const NumericMatrix& features
    ) const override;

    std::size_t node_count() const noexcept;
    std::size_t leaf_count() const noexcept;
    std::size_t depth() const noexcept;

    std::vector<CartSplitInfo> splits(
        std::size_t limit = 5
    ) const;

private:
    CartTree tree_;
};

class CartClassifier final : public IClassifier {
public:
    explicit CartClassifier(
        CartParameters parameters = {},
        FeatureSearchParams feature_search = {},
        std::uint32_t random_state = 0
    );

    void fit(
        const NumericMatrix& features,
        std::span<const double> target
    ) override;

    void fit_with_counts(
        const NumericMatrix& features,
        std::span<const double> target,
        std::span<const std::size_t> counts
    );

    std::vector<double> predict(
        const NumericMatrix& features
    ) const override;

    NumericMatrix predict_proba(
        const NumericMatrix& features
    ) const override;

    std::span<const double> classes() const override;

    std::size_t node_count() const noexcept;
    std::size_t leaf_count() const noexcept;
    std::size_t depth() const noexcept;

    std::vector<CartSplitInfo> splits(
        std::size_t limit = 5
    ) const;

private:
    CartTree tree_;
};


