#include "metrics.hpp"

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    double actual,
    double expected,
    const std::string& message
) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > 1e-10) {
        throw std::runtime_error(message);
    }
}

void require_throws(
    const std::function<void()>& action,
    const std::string& message
) {
    bool caught = false;

    try {
        action();
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    require(caught, message);
}

void test_regression_metrics() {
    const std::vector<double> actual = {1.0, 2.0, 3.0};
    const std::vector<double> predicted = {2.0, 2.0, 2.0};

    const auto result = lab5::regression_metrics(
        actual, predicted
    );

    require_close(
        result.mse, 2.0 / 3.0,
        "Incorrect MSE"
    );

    require_close(
        result.mae, 2.0 / 3.0,
        "Incorrect MAE"
    );

    require_close(
        result.r2, 0.0,
        "Incorrect R2"
    );

    std::cout << "[PASS] Regression metrics\n";
}

void test_negative_r2() {
    const std::vector<double> actual = {1.0, 2.0, 3.0};
    const std::vector<double> predicted = {3.0, 3.0, 3.0};

    require_close(
        lab5::r2_score(actual, predicted),
        -1.5,
        "Negative R2 calculated incorrectly"
    );

    std::cout << "[PASS] Negative R2\n";
}

void test_constant_target() {
    const std::vector<double> actual = {5.0, 5.0, 5.0};

    const std::vector<double> perfect = {5.0, 5.0, 5.0};
    const std::vector<double> imperfect = {4.0, 5.0, 6.0};

    require_close(
        lab5::r2_score(actual, perfect),
        1.0,
        "Constant perfect target should have R2=1"
    );

    require_close(
        lab5::r2_score(actual, imperfect),
        0.0,
        "Constant imperfect target should have R2=0"
    );

    std::cout << "[PASS] Constant target\n";
}

void test_classification_metrics() {
    const std::vector<double> actual = {0, 0, 1, 1};
    const std::vector<double> predicted = {0, 1, 1, 1};
    const std::vector<double> classes = {0, 1};

    const auto result = lab5::classification_metrics(
        actual, predicted, classes
    );

    require_close(
        result.accuracy,
        0.75,
        "Incorrect accuracy"
    );

    require_close(
        result.precision,
        5.0 / 6.0,
        "Incorrect macro-precision"
    );

    require_close(
        result.recall,
        0.75,
        "Incorrect macro-recall"
    );

    // Class 0: F1 = 2/3.
    // Class 1: F1 = 4/5.
    // Macro-F1 = ((2/3) + (4/5)) / 2 = 11/15.
    require_close(
        result.f1,
        11.0 / 15.0,
        "Incorrect macro-F1"
    );

    std::cout << "[PASS] Classification metrics\n";
}

void test_fixed_class_list() {
    const std::vector<double> actual = {0, 1};
    const std::vector<double> predicted = {0, 0};

    // Class 2 is part of the task, but absent from
    // this particular evaluation sample.
    const std::vector<double> classes = {0, 1, 2};

    const auto result = lab5::classification_metrics(
        actual, predicted, classes
    );

    require_close(
        result.accuracy,
        0.5,
        "Incorrect accuracy with absent class"
    );

    require_close(
        result.precision,
        1.0 / 6.0,
        "Absent classes changed macro-precision"
    );

    require_close(
        result.recall,
        1.0 / 3.0,
        "Absent classes changed macro-recall"
    );

    require_close(
        result.f1,
        2.0 / 9.0,
        "Absent classes changed macro-F1"
    );

    std::cout << "[PASS] Fixed class list\n";
}

void test_class_imbalance() {
    // A model that always predicts the majority class.
    std::vector<double> actual(100, 0.0);
    actual.back() = 1.0;

    const std::vector<double> predicted(100, 0.0);
    const std::vector<double> classes = {0, 1};

    const auto result = lab5::classification_metrics(
        actual, predicted, classes
    );

    require_close(
        result.accuracy,
        0.99,
        "Incorrect imbalanced accuracy"
    );

    require(
        result.f1 < 0.5,
        "Macro-F1 should expose the ignored minority class"
    );

    std::cout << "[PASS] Class imbalance\n";
}

void test_scoring_names_and_directions() {
    using lab5::Scoring;

    require(
        lab5::parse_scoring("mse") == Scoring::MSE,
        "MSE parsing failed"
    );

    require(
        lab5::parse_scoring("mae") == Scoring::MAE,
        "MAE parsing failed"
    );

    require(
        lab5::parse_scoring("r2") == Scoring::R2,
        "R2 parsing failed"
    );

    require(
        lab5::parse_scoring("accuracy") == Scoring::Accuracy,
        "Accuracy parsing failed"
    );

    require(
        lab5::parse_scoring("precision") == Scoring::Precision,
        "Precision parsing failed"
    );

    require(
        lab5::parse_scoring("recall") == Scoring::Recall,
        "Recall parsing failed"
    );

    require(
        lab5::parse_scoring("f1") == Scoring::F1,
        "F1 parsing failed"
    );

    require(
        lab5::scoring_name(Scoring::F1) == "f1",
        "Incorrect scoring name"
    );

    require(
        !lab5::higher_is_better(Scoring::MSE),
        "MSE must be minimized"
    );

    require(
        lab5::higher_is_better(Scoring::F1),
        "F1 must be maximized"
    );

    require(
        lab5::is_better_score(0.1, 0.2, Scoring::MSE),
        "Lower MSE should win"
    );

    require(
        lab5::is_better_score(0.9, 0.8, Scoring::F1),
        "Higher F1 should win"
    );

    require(
        !lab5::is_better_score(0.8, 0.8, Scoring::F1),
        "Ties must preserve the first candidate"
    );

    lab5::validate_scoring(Scoring::MSE, false);
    lab5::validate_scoring(Scoring::F1, true);

    require_throws(
        [] {
            lab5::validate_scoring(
                lab5::Scoring::F1, false
            );
        },
        "Classification metric accepted for regression"
    );

    std::cout << "[PASS] Scoring selection\n";
}

void test_scoring_dispatch() {
    const std::vector<double> actual = {1, 2, 3};
    const std::vector<double> predicted = {2, 2, 2};

    require_close(
        lab5::regression_scoring(
            lab5::Scoring::MSE, actual, predicted
        ),
        2.0 / 3.0,
        "Regression scoring dispatch failed"
    );

    const std::vector<double> labels = {0, 0, 1, 1};
    const std::vector<double> guesses = {0, 1, 1, 1};
    const std::vector<double> classes = {0, 1};

    require_close(
        lab5::classification_scoring(
            lab5::Scoring::F1,
            labels,
            guesses,
            classes
        ),
        11.0 / 15.0,
        "Classification scoring dispatch failed"
    );

    std::cout << "[PASS] Scoring dispatch\n";
}

void test_invalid_inputs() {
    require_throws(
        [] {
            const std::vector<double> empty;
            (void)lab5::mean_squared_error(empty, empty);
        },
        "Empty input was accepted"
    );

    require_throws(
        [] {
            const std::vector<double> a = {1, 2};
            const std::vector<double> b = {1};
            (void)lab5::mean_absolute_error(a, b);
        },
        "Mismatched lengths were accepted"
    );

    require_throws(
        [] {
            const std::vector<double> a = {1};
            (void)lab5::r2_score(a, a);
        },
        "R2 accepted a single object"
    );

    require_throws(
        [] {
            const std::vector<double> a = {0, 1};
            const std::vector<double> b = {0, 2};
            const std::vector<double> classes = {0, 1};

            (void)lab5::classification_metrics(
                a, b, classes
            );
        },
        "Unknown predicted class was accepted"
    );

    require_throws(
        [] {
            const std::vector<double> a = {0, 1};
            const std::vector<double> classes = {0, 0};

            (void)lab5::classification_metrics(
                a, a, classes
            );
        },
        "Duplicate class labels were accepted"
    );

    require_throws(
        [] {
            (void)lab5::parse_scoring("unknown");
        },
        "Unknown scoring name was accepted"
    );

    std::cout << "[PASS] Invalid inputs\n";
}

} // namespace

int main() {
    try {
        test_regression_metrics();
        test_negative_r2();
        test_constant_target();
        test_classification_metrics();
        test_fixed_class_list();
        test_class_imbalance();
        test_scoring_names_and_directions();
        test_scoring_dispatch();
        test_invalid_inputs();

        std::cout << "All Lab 5 metrics tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}






