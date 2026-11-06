#include "cart.hpp"
#include "decision_stump.hpp"
#include "ml_exceptions.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void close(double actual, double expected,
           const std::string& message) {
    check(std::abs(actual - expected) < 1e-9, message);
}

template <class Function>
void expect_exception(Function function,
                      const std::string& message) {
    bool thrown = false;
    try {
        function();
    } catch (const std::exception&) {
        thrown = true;
    }
    check(thrown, message);
}

NumericMatrix make_matrix(
    std::size_t rows,
    std::size_t cols,
    const std::vector<double>& row_major_values,
    MatrixLayout layout = MatrixLayout::RowMajor
) {
    check(row_major_values.size() == rows * cols,
          "Invalid test fixture");

    NumericMatrix result;
    result.rows = rows;
    result.cols = cols;
    result.layout = layout;
    result.values.resize(rows * cols);

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const std::size_t offset =
                layout == MatrixLayout::RowMajor
                    ? i * cols + j
                    : j * rows + i;

            result.values[offset] =
                row_major_values[i * cols + j];
        }
    }

    return result;
}

void test_regression_depth() {
    const auto x = make_matrix(
        8, 1, {1, 2, 3, 4, 5, 6, 7, 8}
    );
    const std::vector<double> y =
        {2, 2, 3, 3, 10, 10, 12, 12};

    CartRegressor constant({0, 2, 1});
    constant.fit(x, y);

    check(constant.node_count() == 1,
          "Depth 0 should have one node");
    check(constant.leaf_count() == 1,
          "Depth 0 should have one leaf");
    check(constant.depth() == 0,
          "Depth 0 actual depth");

    for (double value : constant.predict(x)) {
        close(value, 6.75, "Root mean");
    }

    CartRegressor tree({3, 2, 1});
    tree.fit(x, y);

    check(tree.depth() > 1,
          "Regression tree must grow beyond a stump");
    check(tree.node_count() == 7,
          "Expected seven regression nodes");
    check(tree.leaf_count() == 4,
          "Expected four regression leaves");

    const auto predictions = tree.predict(x);
    for (std::size_t i = 0; i < y.size(); ++i) {
        close(predictions[i], y[i],
              "Regression training prediction");
    }
}

void test_classification_probabilities() {
    const auto x = make_matrix(
        6, 1, {1, 2, 3, 4, 5, 6}
    );
    const std::vector<double> y =
        {10, 10, 42, 42, 100, 100};

    CartClassifier model({3, 2, 1});

    expect_exception(
        [&] { (void)model.classes(); },
        "classes() before fit must fail"
    );
    expect_exception(
        [&] { (void)model.predict(x); },
        "predict() before fit must fail"
    );

    model.fit(x, y);

    const auto classes = model.classes();
    check(classes.size() == 3, "Three classes expected");
    close(classes[0], 10, "First class");
    close(classes[1], 42, "Second class");
    close(classes[2], 100, "Third class");

    const auto predictions = model.predict(x);
    const auto probabilities = model.predict_proba(x);

    check(probabilities.rows == x.rows,
          "Probability row count");
    check(probabilities.cols == 3,
          "Probability column count");

    for (std::size_t i = 0; i < x.rows; ++i) {
        close(predictions[i], y[i],
              "Classification prediction");

        double sum = 0.0;
        for (std::size_t j = 0; j < classes.size(); ++j) {
            const double p = probabilities(i, j);
            check(p >= 0.0 && p <= 1.0,
                  "Invalid probability");
            sum += p;
        }
        close(sum, 1.0, "Probabilities must sum to one");
    }

    CartClassifier root_only({0, 2, 1});
    root_only.fit(x, y);
    const auto root_proba = root_only.predict_proba(x);
    for (std::size_t j = 0; j < 3; ++j) {
        close(root_proba(0, j), 1.0 / 3.0,
              "Root class proportions");
    }

    // Equal frequencies select the smallest class label.
    close(root_only.predict(x)[0], 10,
          "Classification tie-breaking");
}

void test_depth_one_matches_stump() {
    const auto x = make_matrix(
        8, 2,
        {
            1, 8,
            2, 7,
            3, 6,
            4, 5,
            5, 4,
            6, 3,
            7, 2,
            8, 1
        }
    );

    const std::vector<double> regression_y =
        {2, 2, 3, 3, 10, 10, 12, 12};
    const std::vector<double> classification_y =
        {0, 0, 0, 1, 1, 1, 1, 1};

    for (const auto criterion :
         {SplitCriterion::Variance, SplitCriterion::Gini}) {
        const auto& y =
            criterion == SplitCriterion::Variance
                ? regression_y
                : classification_y;

        DecisionStump stump(criterion);
        stump.fit(x, y);

        std::vector<double> cart_predictions;

        if (criterion == SplitCriterion::Variance) {
            CartRegressor cart({1, 2, 1});
            cart.fit(x, y);
            cart_predictions = cart.predict(x);
        } else {
            CartClassifier cart({1, 2, 1});
            cart.fit(x, y);
            cart_predictions = cart.predict(x);
        }

        const auto stump_predictions = stump.predict(x);
        check(stump_predictions.size() ==
                  cart_predictions.size(),
              "Stump and CART result sizes");

        for (std::size_t i = 0;
             i < stump_predictions.size(); ++i) {
            close(cart_predictions[i], stump_predictions[i],
                  "Depth-one CART must match stump");
        }
    }
}

void test_layout_equivalence() {
    const std::vector<double> values = {
        1, 8, 2, 7, 3, 6, 4, 5,
        5, 4, 6, 3, 7, 2, 8, 1
    };
    const auto row_major = make_matrix(
        8, 2, values, MatrixLayout::RowMajor
    );
    const auto column_major = make_matrix(
        8, 2, values, MatrixLayout::ColumnMajor
    );

    const std::vector<double> labels =
        {0, 0, 0, 1, 1, 1, 1, 1};

    CartClassifier first({3, 2, 1});
    CartClassifier second({3, 2, 1});

    first.fit(row_major, labels);
    second.fit(column_major, labels);

    const auto a = first.predict(row_major);
    const auto b = second.predict(column_major);

    check(a.size() == b.size(), "Layout result sizes");

    for (std::size_t i = 0; i < a.size(); ++i) {
        close(a[i], b[i], "Layout prediction mismatch");
    }

    check(first.node_count() == second.node_count(),
          "Layout node count mismatch");
}

void test_refit_and_empty_prediction() {
    const auto x = make_matrix(4, 1, {1, 2, 3, 4});
    const std::vector<double> first_y = {0, 0, 1, 1};
    const std::vector<double> second_y = {42, 42, 42, 42};

    CartClassifier model({3, 2, 1});
    model.fit(x, first_y);
    check(model.classes().size() == 2,
          "First fit classes");

    model.fit(x, second_y);
    check(model.classes().size() == 1,
          "Refit must replace classes");
    check(model.node_count() == 1,
          "Single-class tree must have one node");

    for (double value : model.predict(x)) {
        close(value, 42, "Refit prediction");
    }

    NumericMatrix empty;
    empty.rows = 0;
    empty.cols = 1;

    check(model.predict(empty).empty(),
          "Empty prediction vector");

    const auto proba = model.predict_proba(empty);
    check(proba.rows == 0 && proba.cols == 1,
          "Empty probability matrix dimensions");
    check(proba.values.empty(),
          "Empty probability matrix values");
}

void test_invalid_inputs() {
    const auto x = make_matrix(3, 1, {1, 2, 3});
    const std::vector<double> y = {0, 1, 1};

    expect_exception(
        [] { CartRegressor model({2, 1, 1}); },
        "min_samples_split=1 must fail"
    );
    expect_exception(
        [] { CartClassifier model({2, 2, 0}); },
        "min_samples_leaf=0 must fail"
    );

    CartClassifier model;
    expect_exception(
        [&] {
            model.fit(x, std::vector<double>{0, 1});
        },
        "Mismatched target size"
    );

    NumericMatrix bad = x;
    bad.values[0] = std::numeric_limits<double>::quiet_NaN();

    expect_exception(
        [&] { model.fit(bad, y); },
        "NaN in training data"
    );

    model.fit(x, y);

    expect_exception(
        [&] { (void)model.predict(bad); },
        "NaN in prediction data"
    );

    const auto wrong_features = make_matrix(
        3, 2, {1, 2, 3, 4, 5, 6}
    );

    expect_exception(
        [&] { (void)model.predict(wrong_features); },
        "Wrong prediction feature count"
    );
}

} // namespace

int main() {
    try {
        test_regression_depth();
        test_classification_probabilities();
        test_depth_one_matches_stump();
        test_layout_equivalence();
        test_refit_and_empty_prediction();
        test_invalid_inputs();

        std::cout << "[PASS] CART tests\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}





