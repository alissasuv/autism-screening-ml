#include "lab5_preprocessing.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool close(double a, double b) {
    return std::abs(a - b) < 1e-10;
}

RawDataset training_data() {
    RawDataset data;

    data.column_names = {"age", "color", "target"};
    data.row_count = 4;

    data.columns = {
        {"10", "20", "30", std::nullopt},
        {"red", "blue", "red", std::nullopt},
        {"NO", "YES", "NO", "YES"}
    };

    return data;
}

RawDataset validation_data() {
    RawDataset data;

    data.column_names = {"age", "color", "target"};
    data.row_count = 2;

    data.columns = {
        {"1000", std::nullopt},
        {"green", "blue"},
        {"YES", "NO"}
    };

    return data;
}

void test_train_only_statistics(
    const PreprocessingOptions& options
) {
    const auto fitted = lab5::fit_raw_preprocessing(
        training_data(), "target", options
    );

    require(
        fitted.column_types[0] == FeatureType::Numerical,
        "Age should be numeric"
    );

    require(
        fitted.column_types[1] == FeatureType::Categorical,
        "Color should be categorical"
    );

    const auto& stats = fitted.state.numeric_statistics[0];

    require(stats.has_value(), "Missing age statistics");
    require(close(stats->mean, 20.0), "Wrong train mean");
    require(close(stats->q50, 20.0), "Wrong train median");

    const auto transformed =
        lab5::transform_raw_preprocessing(
            validation_data(), fitted
        );

    require(transformed.rows == 2, "Wrong validation rows");

    // Validation age=1000 must not change fitted statistics.
    require(
        close(fitted.state.numeric_statistics[0]->mean, 20.0),
        "Data leakage detected"
    );

    // The missing age is replaced with median=20.
    // Standard scaling maps this value to zero.
    require(
        close(transformed(1, 0), 0.0),
        "Incorrect imputation or scaling"
    );

    std::cout << "[PASS] Training statistics\n";
}

void test_one_hot(
    const PreprocessingOptions& options
) {
    const auto fitted = lab5::fit_raw_preprocessing(
        training_data(), "target", options
    );

    const auto categories_before = fitted.state.categories[1];

    const auto result = lab5::transform_raw_preprocessing(
        validation_data(), fitted
    );

    // age + blue, red, __MISSING__
    require(result.cols == 4, "Wrong feature count");

    // Unknown category "green" must produce all zeros.
    for (std::size_t col = 1; col < result.cols; ++col) {
        require(
            close(result(0, col), 0.0),
            "Unknown category was not encoded as zeros"
        );
    }

    require(
        fitted.state.categories[1] == categories_before,
        "Validation changed the category dictionary"
    );

    std::cout << "[PASS] One-hot vocabulary\n";
}

void test_target(
    const PreprocessingOptions& options
) {
    const auto fitted = lab5::fit_raw_preprocessing(
        training_data(), "target", options
    );

    const auto result = lab5::transform_raw_preprocessing(
        validation_data(), fitted
    );

    require(
        result.target_values ==
            std::vector<std::string>{"YES", "NO"},
        "Custom target column was not preserved"
    );

    std::cout << "[PASS] Custom target\n";
}

void test_regression_target(
    const PreprocessingOptions& options
) {
    RawDataset data;

    data.column_names = {"x", "price"};
    data.row_count = 4;

    data.columns = {
        {"0", "1", "2", "3"},
        {"1.5", "2.5", "3.5", "4.5"}
    };

    const auto fitted = lab5::fit_raw_preprocessing(
        data, "price", options
    );

    const auto output = lab5::transform_raw_preprocessing(
        data, fitted
    );

    require(output.rows == 4, "Wrong row count");
    require(output.cols == 1, "Target became a feature");
    require(
        output.target_values ==
            std::vector<std::string>{
                "1.5", "2.5", "3.5", "4.5"
            },
        "Regression target changed"
    );

    std::cout << "[PASS] Regression target\n";
}

void test_schema_validation(
    const PreprocessingOptions& options
) {
    const auto fitted = lab5::fit_raw_preprocessing(
        training_data(), "target", options
    );

    auto invalid = validation_data();
    invalid.column_names[0] = "different_age";

    bool rejected = false;

    try {
        (void)lab5::transform_raw_preprocessing(
            invalid, fitted
        );
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    require(rejected, "Invalid schema was accepted");

    std::cout << "[PASS] Schema validation\n";
}

void test_repeated_transform(
    const PreprocessingOptions& options
) {
    const auto fitted = lab5::fit_raw_preprocessing(
        training_data(), "target", options
    );

    const auto first = lab5::transform_raw_preprocessing(
        validation_data(), fitted
    );

    const auto second = lab5::transform_raw_preprocessing(
        validation_data(), fitted
    );

    require(
        first.column_names == second.column_names,
        "Column order changed"
    );

    require(
        first.values == second.values,
        "Repeated transform changed its results"
    );

    std::cout << "[PASS] Repeated transform\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Expected YAML config path");

        const auto options =
            lab5::load_preprocessing_config(argv[1]);

        require(
            options.imputation == NumericImputation::Median,
            "Wrong YAML numeric imputation"
        );

        require(
            options.scaling == NumericScaling::Standard,
            "Wrong YAML numeric scaling"
        );

        require(
            options.encoding == CategoryEncoding::OneHot,
            "Wrong YAML encoding"
        );

        test_train_only_statistics(options);
        test_one_hot(options);
        test_target(options);
        test_regression_target(options);
        test_schema_validation(options);
        test_repeated_transform(options);

        std::cout
            << "All Lab 5 preprocessing tests passed.\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}


