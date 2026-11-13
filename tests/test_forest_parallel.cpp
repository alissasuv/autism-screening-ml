#include "forest_parallel.hpp"
#include "random_forest.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

NumericMatrix make_data(MatrixLayout layout) {
    NumericMatrix x;

    x.rows = 24;
    x.cols = 3;
    x.layout = layout;
    x.values.resize(x.rows * x.cols);

    for (std::size_t r = 0; r < x.rows; ++r) {
        for (std::size_t c = 0; c < x.cols; ++c) {
            const std::size_t index =
                layout == MatrixLayout::RowMajor
                    ? r * x.cols + c
                    : c * x.rows + r;

            x.values[index] =
                static_cast<double>(
                    (r * (c + 3) + c * 7) % 17
                );
        }
    }

    return x;
}

void test_layout(MatrixLayout layout) {
    const auto x = make_data(layout);

    std::vector<double> regression_y(x.rows);
    std::vector<double> classification_y(x.rows);

    for (std::size_t i = 0; i < x.rows; ++i) {
        regression_y[i] =
            static_cast<double>((i * 11) % 19);

        classification_y[i] =
            static_cast<double>(i % 3);
    }

    RandomForestParameters p;
    p.n_estimators = 11;
    p.random_state = 827;
    p.max_features = {FeatureMode::Count, 2};
    p.tree.max_depth = 4;
    p.n_jobs = 1;

    // Sequential regression reference.
    RandomForestRegressor sequential(p);
    sequential.fit(x, regression_y);

    const auto expected_regression =
        sequential.predict(x);

    const auto expected_nodes =
        sequential.tree_node_counts();

    // Parallel regression must be identical.
    p.n_jobs = 4;

    RandomForestRegressor parallel(p);
    parallel.fit(x, regression_y);

    check(
        parallel.predict(x) == expected_regression,
        "Regression differs by n_jobs"
    );

    check(
        parallel.tree_node_counts() == expected_nodes,
        "Regression tree plans differ"
    );

    parallel.fit(x, regression_y);

    check(
        parallel.predict(x) == expected_regression,
        "Parallel refit changed regression"
    );

    // Sequential classification reference.
    p.max_features = {FeatureMode::Sqrt, 1};
    p.n_jobs = 1;

    RandomForestClassifier sequential_classifier(p);
    sequential_classifier.fit(x, classification_y);

    const auto expected_probabilities =
        sequential_classifier.predict_proba(x).values;

    const auto expected_classes =
        sequential_classifier.predict(x);

    // Parallel classification must be identical.
    p.n_jobs = 4;

    RandomForestClassifier parallel_classifier(p);
    parallel_classifier.fit(x, classification_y);

    check(
        parallel_classifier.predict_proba(x).values ==
            expected_probabilities,
        "Classification probabilities differ by n_jobs"
    );

    check(
        parallel_classifier.predict(x) ==
            expected_classes,
        "Classification labels differ by n_jobs"
    );

    parallel_classifier.fit(x, classification_y);

    check(
        parallel_classifier.predict_proba(x).values ==
            expected_probabilities,
        "Parallel refit changed classification"
    );

    // Empty prediction batches.
    NumericMatrix empty = x;
    empty.rows = 0;
    empty.values.clear();

    check(
        parallel.predict(empty).empty(),
        "Empty regression predictions"
    );

    const auto proba =
        parallel_classifier.predict_proba(empty);

    check(
        proba.rows == 0 &&
        proba.cols == 3 &&
        proba.values.empty(),
        "Empty classification probabilities"
    );
}

void test_worker_exceptions() {
    std::vector<std::size_t> seen(7, 0);

    bool caught = false;

    try {
        forest_parallel::for_each_index(
            7,
            3,
            [&](std::size_t i) {
                seen[i] = 1;

                if (i == 4) {
                    throw std::runtime_error(
                        "worker exception"
                    );
                }
            }
        );
    } catch (const std::runtime_error& error) {
        caught =
            std::string(error.what()) ==
            "worker exception";
    }

    check(
        caught,
        "Worker exception was not propagated"
    );

    for (const auto value : seen) {
        check(value == 1, "Worker missed a task");
    }

    // n_jobs must not create extra work.
    std::vector<std::size_t> once(1, 0);

    forest_parallel::for_each_index(
        1,
        8,
        [&](std::size_t) {
            ++once[0];
        }
    );

    check(
        once[0] == 1,
        "Too many tasks for one tree"
    );
}

} // namespace

int main() {
    try {
        test_layout(MatrixLayout::RowMajor);
        test_layout(MatrixLayout::ColumnMajor);
        test_worker_exceptions();

        std::cout << "Forest parallel tests passed\n";
    } catch (const std::exception& error) {
        std::cerr
            << "Forest parallel tests failed: "
            << error.what()
            << '\n';

        return 1;
    }

    return 0;
}




