#pragma once

#include "model.hpp"
#include "split_finder.hpp"

#include <cstddef>
#include <span>
#include <vector>

class DecisionStump : public IModel {
public:
    DecisionStump(
        SplitCriterion criterion,
        std::size_t min_samples_leaf = 1
    );

    void fit(
        const NumericMatrix& features,
        std::span<const double> target
    ) override;

    std::vector<double> predict(
        const NumericMatrix& features
    ) const override;

    std::size_t split_feature() const;
    double split_threshold() const;

    double left_prediction() const;
    double right_prediction() const;
    bool has_split() const;

private:
    SplitCriterion criterion_;
    std::size_t min_samples_leaf_;

    SplitFinder split_finder_;

    bool fitted_ = false;
    bool has_split_ = false;

    std::size_t feature_count_ = 0;
    std::size_t split_feature_ = 0;
    double split_threshold_ = 0.0;

    double left_prediction_ = 0.0;
    double right_prediction_ = 0.0;
    double constant_prediction_ = 0.0;

    double calculate_prediction(
        std::span<const double> target,
        const std::vector<std::size_t>& indices
    ) const;
};
