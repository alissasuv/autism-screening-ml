#pragma once

#include "cart.hpp"
#include "model.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

struct RandomForestParameters {
    std::size_t n_estimators = 100;
    bool bootstrap = true;
    std::optional<std::size_t> max_samples = std::nullopt;

    FeatureSearchParams max_features = {
        FeatureMode::All, 1
    };

    CartParameters tree = {
        std::nullopt, 2, 1
    };

    std::uint32_t random_state = 0;
    std::size_t n_jobs = 1;
};

class RandomForestRegressor final : public IRegressor {
public:
    explicit RandomForestRegressor(
        RandomForestParameters parameters = {}
    );

    void fit(
        const NumericMatrix& features,
        std::span<const double> target
    ) override;

    std::vector<double> predict(
        const NumericMatrix& features
    ) const override;

    std::size_t tree_count() const noexcept;

    std::vector<std::size_t> tree_node_counts() const;

    std::vector<std::size_t> tree_depths() const;

private:
    RandomForestParameters parameters_;

    std::vector<std::unique_ptr<CartRegressor>> trees_;
};








