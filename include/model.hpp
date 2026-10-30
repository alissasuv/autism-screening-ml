#pragma once

#include "preprocessing.hpp"

#include <span>
#include <vector>

class IModel {
public:
    virtual ~IModel() = default;

    virtual void fit(
        const NumericMatrix& features,
        std::span<const double> target
    ) = 0;

    virtual std::vector<double> predict(
        const NumericMatrix& features
    ) const = 0;
};
