#include "cart.hpp"
#include "ml_exceptions.hpp"

#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

void check(bool ok, const char* message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

void close(double a, double b, const char* message) {
    check(std::abs(a - b) < 1e-10, message);
}

NumericMatrix matrix(MatrixLayout layout) {
    NumericMatrix x;

    x.rows = 4;
    x.cols = 2;
    x.layout = layout;
    x.values.resize(8);

    const double values[4][2] = {
        {0, 9},
        {1, 9},
        {2, 9},
        {3, 9}
    };

    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 2; ++c) {
            const std::size_t index =
                layout == MatrixLayout::RowMajor
                    ? r * 2 + c
                    : c * 4 + r;

            x.values[index] = values[r][c];
        }
    }

    return x;
}

void test(MatrixLayout layout) {
    const auto x = matrix(layout);

    const std::vector<double> y = {
        0, 0, 10, 10
    };

    const std::vector<std::size_t> weights = {
        2, 1, 3, 0
    };

    // Weighted regression mean.
    CartRegressor reg({0, 2, 1});

    reg.fit_with_counts(x, y, weights);

    for (const auto value : reg.predict(x)) {
        close(value, 5.0, "Weighted mean");
    }

    // Original fit remains available.
    reg.fit(x, y);

    for (const auto value : reg.predict(x)) {
        close(value, 5.0, "Unweighted mean");
    }

    // Weighted class probabilities.
    CartClassifier cls({0, 2, 1});

    cls.fit_with_counts(x, y, weights);

    const auto p = cls.predict_proba(x);

    check(
        p.rows == 4 && p.cols == 2,
        "Probability matrix shape"
    );

    for (std::size_t i = 0; i < 4; ++i) {
        close(p(i, 0), 0.5, "Class 0 probability");
        close(p(i, 1), 0.5, "Class 10 probability");
    }

    check(
        cls.classes().size() == 2,
        "Classes available"
    );

    // Unlimited depth and random feature sampling.
    CartRegressor full(
        {std::nullopt, 2, 1},
        {FeatureMode::Count, 1},
        42
    );

    full.fit_with_counts(x, y, weights);

    check(
        full.depth() == 1,
        "Unlimited depth tree should grow"
    );

    check(
        full.splits(1).at(0).feature == 0,
        "Skip constant feature"
    );

    // Repeated fit must reset RNG.
    const auto old = full.predict(x);

    full.fit_with_counts(x, y, weights);

    check(
        full.predict(x) == old,
        "Repeat fit must be deterministic"
    );

    // A missing bootstrap class is not synthesized.
    CartClassifier missing({0, 2, 1});

    const std::vector<std::size_t> missing_counts = {
        0, 0, 1, 1
    };

    missing.fit_with_counts(x, y, missing_counts);

    check(
        missing.classes().size() == 1,
        "Absent class not synthesized"
    );

    const auto one_class_proba = missing.predict_proba(x);

    close(
        one_class_proba(0, 0),
        1.0,
        "Single-class probability"
    );

    // All-zero bootstrap counts are invalid.
    bool threw = false;

    try {
        const std::vector<std::size_t> zero = {
            0, 0, 0, 0
        };

        reg.fit_with_counts(x, y, zero);
    } catch (const InvalidModelParameterError&) {
        threw = true;
    }

    check(
        threw,
        "Reject all-zero multiplicities"
    );
}

} // namespace

int main() {
    try {
        test(MatrixLayout::RowMajor);
        test(MatrixLayout::ColumnMajor);

        std::cout << "Weighted CART tests passed\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr
            << "Weighted CART tests failed: "
            << e.what()
            << '\n';

        return 1;
    }
}


