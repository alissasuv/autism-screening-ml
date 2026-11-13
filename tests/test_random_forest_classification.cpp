#include "random_forest.hpp"
#include "ml_exceptions.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
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

    const double data[6][2] = {
        {0, 0},
        {1, 1},
        {2, 0},
        {3, 1},
        {4, 0},
        {5, 1}
    };

    for (std::size_t r = 0; r < x.rows; ++r) {
        for (std::size_t c = 0; c < x.cols; ++c) {
            x.values[
                layout == MatrixLayout::RowMajor
                    ? r * x.cols + c
                    : c * x.rows + r
            ] = data[r][c];
        }
    }

    return x;
}

void test(MatrixLayout layout) {
    const auto x = make_matrix(layout);

    const std::vector<double> y = {
        10, 10, 20, 20, 30, 30
    };

    auto parameters = classifier_default_parameters();

    parameters.n_estimators = 13;
    parameters.max_samples = 1;
    parameters.random_state = 17;

    RandomForestClassifier forest(parameters);

    bool threw = false;

    try {
        (void)forest.predict(x);
    } catch (const InvalidModelStateError&) {
        threw = true;
    }

    require(threw, "predict before fit");

    forest.fit(x, y);

    require(
        forest.tree_count() == 13,
        "tree count"
    );

    require(
        forest.tree_node_counts().size() == 13,
        "node counts"
    );

    require(
        forest.tree_depths().size() == 13,
        "depths"
    );

    const auto classes = forest.classes();

    require(
        classes.size() == 3 &&
        classes[0] == 10 &&
        classes[1] == 20 &&
        classes[2] == 30,
        "global class order"
    );

    const auto probabilities = forest.predict_proba(x);

    require(
        probabilities.rows == x.rows &&
        probabilities.cols == 3,
        "probability shape"
    );

    const auto predictions = forest.predict(x);

    require(
        predictions.size() == x.rows,
        "prediction shape"
    );

    for (std::size_t r = 0; r < x.rows; ++r) {
        double sum = 0.0;
        std::size_t argmax = 0;

        for (std::size_t c = 0; c < 3; ++c) {
            require(
                probabilities(r, c) >= 0.0 &&
                probabilities(r, c) <= 1.0,
                "valid class probability"
            );

            sum += probabilities(r, c);

            if (probabilities(r, c) >
                probabilities(r, argmax)) {
                argmax = c;
            }
        }

        require(
            std::abs(sum - 1.0) < 1e-12,
            "probabilities sum to one"
        );

        require(
            predictions[r] == classes[argmax],
            "predict uses probabilities"
        );
    }

    // Repeated fit must give identical results.
    forest.fit(x, y);

    require(
        forest.predict_proba(x).values ==
            probabilities.values,
        "repeated fit deterministic"
    );

    RandomForestClassifier same(parameters);
    same.fit(x, y);

    require(
        same.predict_proba(x).values ==
            probabilities.values,
        "same seed deterministic"
    );

    // A single full-data tree must match ordinary CART.
    parameters.bootstrap = false;
    parameters.max_samples.reset();
    parameters.n_estimators = 1;
    parameters.max_features = {FeatureMode::All, 1};

    RandomForestClassifier one(parameters);
    one.fit(x, y);

    CartClassifier cart(parameters.tree);
    cart.fit(x, y);

    require(
        one.predict(x) == cart.predict(x),
        "single-tree labels"
    );

    require(
        one.predict_proba(x).values ==
            cart.predict_proba(x).values,
        "single-tree probabilities"
    );

    // Empty prediction batches preserve the class count.
    NumericMatrix empty = x;
    empty.rows = 0;
    empty.values.clear();

    require(
        one.predict(empty).empty(),
        "empty labels"
    );

    const auto empty_proba = one.predict_proba(empty);

    require(
        empty_proba.rows == 0 &&
        empty_proba.cols == 3 &&
        empty_proba.values.empty(),
        "empty probability shape"
    );

    // A failed fit must preserve the previous model.
    threw = false;

    try {
        one.fit(
            x,
            std::span<const double>(
                y.data(),
                y.size() - 1
            )
        );
    } catch (const DimensionMismatchError&) {
        threw = true;
    }

    require(
        threw,
        "invalid fit rejected"
    );

    require(
        one.predict_proba(x).values ==
            cart.predict_proba(x).values,
        "failed refit preserves state"
    );

    // A single-class training set is valid.
    const std::vector<double> same_class(6, 7.0);

    RandomForestClassifier single(parameters);
    single.fit(x, same_class);

    require(
        single.classes().size() == 1 &&
        single.classes()[0] == 7.0,
        "one-class vocabulary"
    );

    for (double p : single.predict_proba(x).values) {
        require(
            p == 1.0,
            "one-class probability"
        );
    }
}

} // namespace

int main() {
    try {
        test(MatrixLayout::RowMajor);
        test(MatrixLayout::ColumnMajor);

        std::cout
            << "Random Forest classification tests passed\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "Random Forest classification tests failed: "
            << error.what()
            << '\n';

        return 1;
    }
}



