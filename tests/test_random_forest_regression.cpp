#include "random_forest.hpp"
#include "ml_exceptions.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

NumericMatrix make_matrix(MatrixLayout layout) {
    NumericMatrix x;

    x.rows = 6;
    x.cols = 2;
    x.layout = layout;
    x.values.resize(12);

    for (std::size_t r = 0; r < x.rows; ++r) {
        for (std::size_t c = 0; c < x.cols; ++c) {
            x.values[
                layout == MatrixLayout::RowMajor
                    ? r * x.cols + c
                    : c * x.rows + r
            ] = c == 0
                ? static_cast<double>(r)
                : static_cast<double>(r % 2);
        }
    }

    return x;
}

void test(MatrixLayout layout) {
    const auto x = make_matrix(layout);

    const std::vector<double> y = {
        0, 0, 2, 2, 7, 7
    };

    RandomForestParameters p;
    p.n_estimators = 9;
    p.random_state = 73;
    p.max_features = {FeatureMode::Count, 1};
    p.tree.max_depth = 3;

    RandomForestRegressor forest(p);

    bool before_fit_throws = false;

    try {
        (void)forest.predict(x);
    } catch (const InvalidModelStateError&) {
        before_fit_throws = true;
    }

    expect(
        before_fit_throws,
        "predict before fit must throw"
    );

    forest.fit(x, y);

    expect(
        forest.tree_count() == 9,
        "tree count"
    );

    expect(
        forest.tree_node_counts().size() == 9,
        "node counts"
    );

    expect(
        forest.tree_depths().size() == 9,
        "depths"
    );

    const auto first = forest.predict(x);

    expect(
        first.size() == x.rows,
        "prediction size"
    );

    for (double value : first) {
        expect(
            std::isfinite(value),
            "finite prediction"
        );
    }

    // Repeated fit must reconstruct the same forest.
    forest.fit(x, y);

    expect(
        forest.predict(x) == first,
        "repeat fit reproducibility"
    );

    expect(
        forest.tree_count() == 9,
        "fit replaces rather than appends"
    );

    RandomForestRegressor second(p);
    second.fit(x, y);

    expect(
        second.predict(x) == first,
        "same seed reproducibility"
    );

    // A single full-data tree matches ordinary CART.
    p.n_estimators = 1;
    p.bootstrap = false;
    p.max_features = {FeatureMode::All, 1};

    RandomForestRegressor one(p);
    one.fit(x, y);

    CartRegressor cart(p.tree);
    cart.fit(x, y);

    expect(
        one.predict(x) == cart.predict(x),
        "single tree matches CART"
    );

    // An empty prediction batch is valid.
    NumericMatrix empty = x;
    empty.rows = 0;
    empty.values.clear();

    expect(
        one.predict(empty).empty(),
        "empty prediction batch"
    );

    // Failed refitting preserves old trained trees.
    bool rejected = false;

    try {
        one.fit(
            x,
            std::span<const double>(
                y.data(),
                y.size() - 1
            )
        );
    } catch (const DimensionMismatchError&) {
        rejected = true;
    }

    expect(
        rejected,
        "bad fit rejected"
    );

    expect(
        one.predict(x) == cart.predict(x),
        "failed fit keeps old trees"
    );
}

void invalid_parameters() {
    RandomForestParameters p;
    p.n_estimators = 0;

    bool rejected = false;

    try {
        RandomForestRegressor invalid(p);
    } catch (const InvalidModelParameterError&) {
        rejected = true;
    }

    expect(
        rejected,
        "zero trees rejected"
    );

    p.n_estimators = 1;
    p.bootstrap = false;
    p.max_samples = 2;

    rejected = false;

    try {
        RandomForestRegressor invalid(p);
    } catch (const InvalidModelParameterError&) {
        rejected = true;
    }

    expect(
        rejected,
        "max_samples without bootstrap rejected"
    );
}

} // namespace

int main() {
    try {
        invalid_parameters();

        test(MatrixLayout::RowMajor);
        test(MatrixLayout::ColumnMajor);

        std::cout
            << "Random Forest regression tests passed\n";

    } catch (const std::exception& e) {
        std::cerr
            << "Random Forest regression tests failed: "
            << e.what()
            << '\n';

        return 1;
    }

    return 0;
}







