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

class IRegressor : public IModel {
public:
    ~IRegressor() override = default;
};

class IClassifier : public IModel {
public:
    ~IClassifier() override = default;

    virtual NumericMatrix predict_proba(
        const NumericMatrix& features
    ) const = 0;

    virtual std::span<const double> classes() const = 0;
};
