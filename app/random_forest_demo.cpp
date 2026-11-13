#include "cart.hpp"
#include "csv_reader.hpp"
#include "dataset.hpp"
#include "feature_type.hpp"
#include "preprocessing.hpp"
#include "random_forest.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Timings {
    double fit_ms = 0.0;
    double predict_ms = 0.0;
};

double elapsed_ms(
    Clock::time_point start,
    Clock::time_point finish
) {
    return std::chrono::duration<double, std::milli>(
        finish - start
    ).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());

    const std::size_t middle = values.size() / 2;

    return values.size() % 2 == 1
        ? values[middle]
        : (values[middle - 1] + values[middle]) / 2.0;
}

template <typename MakeModel>
Timings benchmark(
    MakeModel make_model,
    const NumericMatrix& train,
    const std::vector<double>& y_train,
    const NumericMatrix& test
) {
    constexpr std::size_t repeats = 3;

    std::vector<double> fit_times;
    std::vector<double> predict_times;

    for (std::size_t repeat = 0;
         repeat < repeats;
         ++repeat) {

        auto model = make_model();

        const auto begin_fit = Clock::now();

        model.fit(train, y_train);

        const auto end_fit = Clock::now();

        const auto begin_predict = Clock::now();

        const auto predictions = model.predict(test);

        const auto end_predict = Clock::now();

        if (predictions.size() != test.rows) {
            throw std::runtime_error(
                "Unexpected prediction count"
            );
        }

        fit_times.push_back(
            elapsed_ms(begin_fit, end_fit)
        );

        predict_times.push_back(
            elapsed_ms(begin_predict, end_predict)
        );
    }

    return {
        median(fit_times),
        median(predict_times)
    };
}

double accuracy(
    const std::vector<double>& predicted,
    const std::vector<double>& actual
) {
    if (predicted.size() != actual.size() ||
        actual.empty()) {
        throw std::runtime_error(
            "Invalid accuracy inputs"
        );
    }

    std::size_t correct = 0;

    for (std::size_t i = 0;
         i < actual.size();
         ++i) {
        correct += static_cast<std::size_t>(
            predicted[i] == actual[i]
        );
    }

    return static_cast<double>(correct) /
           static_cast<double>(actual.size());
}

double mae(
    const std::vector<double>& predicted,
    const std::vector<double>& actual
) {
    if (predicted.size() != actual.size() ||
        actual.empty()) {
        throw std::runtime_error(
            "Invalid MAE inputs"
        );
    }

    double error = 0.0;

    for (std::size_t i = 0;
         i < actual.size();
         ++i) {
        error += std::abs(
            predicted[i] - actual[i]
        );
    }

    return error /
           static_cast<double>(actual.size());
}

void print_timings(
    const std::string& name,
    Timings times
) {
    std::cout
        << std::left << std::setw(23) << name
        << " fit="
        << std::right << std::setw(10)
        << times.fit_ms
        << " ms, predict="
        << std::setw(10)
        << times.predict_ms
        << " ms\n";
}

void regression_demo() {
    NumericMatrix x;

    x.rows = 12;
    x.cols = 1;
    x.column_names = {"x"};

    x.values = {
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10, 11
    };

    const std::vector<double> y = {
        0, 1, 1, 2, 3, 5,
        8, 8, 9, 10, 11, 12
    };

    CartParameters tree{4, 2, 1};

    CartRegressor cart(tree);
    cart.fit(x, y);

    RandomForestParameters parameters;

    parameters.n_estimators = 25;
    parameters.random_state = 42;
    parameters.tree = tree;
    parameters.n_jobs = 4;

    RandomForestRegressor forest(parameters);
    forest.fit(x, y);

    const auto cart_prediction = cart.predict(x);
    const auto forest_prediction = forest.predict(x);

    std::cout
        << "\nRegression (separate synthetic example)\n"
        << "CART MAE=" << mae(cart_prediction, y)
        << ", Random Forest MAE="
        << mae(forest_prediction, y)
        << " (training-set illustration only)\n";

    for (std::size_t i = 0; i < 4; ++i) {
        std::cout
            << "x=" << x(i, 0)
            << " actual=" << y[i]
            << " CART=" << cart_prediction[i]
            << " forest=" << forest_prediction[i]
            << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string path =
            argc > 1
                ? argv[1]
                : "data/Autism-Adult-Data.csv";

        // Data loading and preprocessing are timed separately.
        const auto begin_preprocessing = Clock::now();

        const auto raw = read_csv(path);

        std::vector<FeatureType> types;
        types.reserve(raw.columns.size());

        for (const auto& column : raw.columns) {
            types.push_back(
                detect_feature_type(column)
            );
        }

        const auto dataset =
            make_typed_dataset(raw, types);

        const auto split =
            split_dataset(dataset, 0.8, 42);

        const auto state = fit_preprocessing(
            split.train,
            PreprocessingOptions{}
        );

        const auto train = transform_preprocessing(
            split.train,
            state
        );

        const auto test = transform_preprocessing(
            split.test,
            state
        );

        const auto y_train =
            encode_target(train.target_values);

        const auto y_test =
            encode_target(test.target_values);

        const auto end_preprocessing = Clock::now();

        // Shared tree constraints for a fair comparison.
        CartParameters tree{7, 2, 2};

        RandomForestParameters parameters;

        parameters.n_estimators = 25;
        parameters.random_state = 42;
        parameters.max_features = {
            FeatureMode::Sqrt, 1
        };
        parameters.tree = tree;
        parameters.n_jobs = 4;

        CartClassifier cart(tree);
        cart.fit(train, y_train);

        RandomForestClassifier forest(parameters);
        forest.fit(train, y_train);

        const auto cart_labels =
            cart.predict(test);

        const auto forest_labels =
            forest.predict(test);

        const auto probabilities =
            forest.predict_proba(test);

        const auto labels = forest.classes();

        std::cout
            << std::fixed
            << std::setprecision(3);

        std::cout
            << "Random Forest - Lab 4 star\n"
            << "Dataset: Autism Screening Adult\n"
            << "Rows: train=" << train.rows
            << ", test=" << test.rows
            << ", features=" << train.cols << '\n'
            << "Preprocessing time (including CSV): "
            << elapsed_ms(
                begin_preprocessing,
                end_preprocessing
            )
            << " ms\n"
            << "CART: max_depth=7, "
            << "min_samples_split=2, "
            << "min_samples_leaf=2\n"
            << "Forest: trees="
            << parameters.n_estimators
            << ", bootstrap=true, "
            << "max_features=Sqrt, seed=42\n";

        std::cout
            << "CART test accuracy="
            << accuracy(cart_labels, y_test)
            << ", nodes=" << cart.node_count()
            << ", depth=" << cart.depth()
            << '\n';

        std::cout
            << "Forest test accuracy="
            << accuracy(forest_labels, y_test)
            << ", average nodes=";

        const auto nodes =
            forest.tree_node_counts();

        const auto depths =
            forest.tree_depths();

        const auto total_nodes = std::accumulate(
            nodes.begin(),
            nodes.end(),
            0.0
        );

        const auto total_depth = std::accumulate(
            depths.begin(),
            depths.end(),
            0.0
        );

        std::cout
            << total_nodes / nodes.size()
            << ", average depth="
            << total_depth / depths.size()
            << '\n';

        std::cout
            << "First predictions and probabilities:\n";

        for (std::size_t row = 0;
             row < std::min<std::size_t>(5, test.rows);
             ++row) {

            std::cout
                << "  actual=" << y_test[row]
                << " CART=" << cart_labels[row]
                << " forest=" << forest_labels[row]
                << " probabilities=[";

            for (std::size_t col = 0;
                 col < labels.size();
                 ++col) {

                if (col != 0) {
                    std::cout << ", ";
                }

                std::cout
                    << labels[col]
                    << ":"
                    << probabilities(row, col);
            }

            std::cout << "]\n";
        }

        // Release build without sanitizers.
        // Three repetitions, median fit and predict times.
        std::cout
            << "\nBenchmark "
            << "(3 repetitions, medians; same data)\n";

        const auto cart_times = benchmark(
            [&] {
                return CartClassifier(tree);
            },
            train,
            y_train,
            test
        );

        print_timings("CART", cart_times);

        parameters.n_jobs = 1;

        const auto single = benchmark(
            [&] {
                return RandomForestClassifier(parameters);
            },
            train,
            y_train,
            test
        );

        print_timings("Forest n_jobs=1", single);

        parameters.n_jobs = 4;

        const auto parallel = benchmark(
            [&] {
                return RandomForestClassifier(parameters);
            },
            train,
            y_train,
            test
        );

        print_timings("Forest n_jobs=4", parallel);

        std::cout
            << "Fit speedup="
            << single.fit_ms / parallel.fit_ms
            << "x, predict speedup="
            << single.predict_ms / parallel.predict_ms
            << "x\n";

        regression_demo();

        std::cout
            << "\nNote: this screening dataset is not "
            << "a clinical diagnostic benchmark.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "Error: "
            << error.what()
            << '\n';

        return 1;
    }
}



