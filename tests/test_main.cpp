#include "binary_io.hpp"
#include "csv_reader.hpp"
#include "dataset.hpp"
#include "feature_type.hpp"
#include "preprocessing.hpp"
#include "statistics.hpp"

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_close(
    double actual,
    double expected,
    const std::string& message
) {
    check(
        std::isfinite(actual) &&
        std::abs(actual - expected) < 1e-9,
        message
    );
}

template <typename Function>
void run_test(
    const std::string& name,
    Function function
) {
    try {
        function();
        ++passed;
        std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception& error) {
        ++failed;
        std::cerr << "[FAIL] " << name
                  << ": " << error.what() << '\n';
    }
}

template <typename Function>
void expect_exception(
    Function function,
    const std::string& message
) {
    bool thrown = false;

    try {
        function();
    } catch (const std::exception&) {
        thrown = true;
    }

    check(thrown, message);
}

void write_text_file(
    const std::string& filename,
    const std::string& content
) {
    std::ofstream file(filename, std::ios::binary);

    check(
        static_cast<bool>(file),
        "Cannot create test file"
    );

    file << content;

    check(
        static_cast<bool>(file),
        "Cannot write test file"
    );
}

Dataset make_example_dataset() {
    Dataset dataset;

    dataset.row_count = 4;
    dataset.column_names = {
        "age",
        "color",
        "Class/ASD"
    };

    NumericColumn age;
    age.values = {
        1.0,
        std::nullopt,
        3.0,
        5.0
    };

    CategoricalColumn color;
    color.values = {
        "red",
        "blue",
        std::nullopt,
        "red"
    };

    CategoricalColumn target;
    target.values = {
        "YES",
        "NO",
        "YES",
        "NO"
    };

    dataset.columns.emplace_back(age);
    dataset.columns.emplace_back(color);
    dataset.columns.emplace_back(target);

    return dataset;
}

std::vector<std::optional<NumericStatistics>>
make_example_statistics(const Dataset& dataset) {
    std::vector<std::optional<NumericStatistics>> result(
        dataset.columns.size()
    );

    const auto& age =
        std::get<NumericColumn>(dataset.columns[0]);

    const auto values = extract_numeric_values(age);

    result[0] = calculate_numeric_statistics(values);

    return result;
}

void test_csv_loading() {
    const std::string filename = "test_valid.csv";

    write_text_file(
        filename,
        "age,color,Class/ASD\n"
        "18,red,YES\n"
        "?,blue,NO\n"
        "25,?,YES\n"
    );

    const auto dataset = read_csv(filename);

    check(dataset.row_count == 3, "Wrong row count");
    check(dataset.columns.size() == 3, "Wrong column count");

    check(
        dataset.column_names[0] == "age",
        "Wrong column name"
    );

    check(
        dataset.columns[0][0].has_value() &&
        *dataset.columns[0][0] == "18",
        "Wrong CSV value"
    );

    check(
        !dataset.columns[0][1].has_value(),
        "Numeric missing value was not detected"
    );

    check(
        !dataset.columns[1][2].has_value(),
        "Categorical missing value was not detected"
    );

    std::filesystem::remove(filename);
}

void test_invalid_csv() {
    const std::string filename = "test_invalid.csv";

    write_text_file(
        filename,
        "age,color\n"
        "18,red\n"
        "25\n"
    );

    expect_exception(
        [&]() {
            (void)read_csv(filename);
        },
        "CSV with inconsistent row lengths was accepted"
    );

    std::filesystem::remove(filename);
}

void test_feature_type() {
    using Cell = std::optional<std::string>;

    const std::vector<Cell> numeric = {
        "1", "2", "3", std::nullopt
    };

    const std::vector<Cell> binary = {
        "0", "1", "0", "1"
    };

    const std::vector<Cell> categorical = {
        "red", "blue", "green"
    };

    check(
        detect_feature_type(numeric) == FeatureType::Numerical,
        "Numeric column classified incorrectly"
    );

    check(
        detect_feature_type(binary) == FeatureType::Categorical,
        "Binary column classified incorrectly"
    );

    check(
        detect_feature_type(categorical) == FeatureType::Categorical,
        "Categorical column classified incorrectly"
    );
}

void test_typed_dataset() {
    RawDataset raw;

    raw.row_count = 3;
    raw.column_names = {"age", "color"};

    raw.columns = {
        {"18", std::nullopt, "25"},
        {"red", "blue", std::nullopt}
    };

    const std::vector<FeatureType> types = {
        FeatureType::Numerical,
        FeatureType::Categorical
    };

    const auto dataset = make_typed_dataset(raw, types);

    check(
        std::holds_alternative<NumericColumn>(dataset.columns[0]),
        "Numeric column has wrong variant type"
    );

    check(
        std::holds_alternative<CategoricalColumn>(dataset.columns[1]),
        "Categorical column has wrong variant type"
    );

    const auto& age =
        std::get<NumericColumn>(dataset.columns[0]);

    check_close(*age.values[0], 18.0, "Wrong numeric value");

    check(
        !age.values[1].has_value(),
        "Typed missing value was lost"
    );
}

void test_numeric_statistics() {
    const std::vector<double> values = {
        1.0, 2.0, 3.0, 4.0, 5.0
    };

    const auto stats = calculate_numeric_statistics(values);

    check_close(stats.min, 1.0, "Wrong minimum");
    check_close(stats.max, 5.0, "Wrong maximum");
    check_close(stats.mean, 3.0, "Wrong mean");
    check_close(stats.variance, 2.0, "Wrong variance");

    check_close(stats.q05, 1.2, "Wrong Q05");
    check_close(stats.q25, 2.0, "Wrong Q25");
    check_close(stats.q50, 3.0, "Wrong median");
    check_close(stats.q75, 4.0, "Wrong Q75");
    check_close(stats.q95, 4.8, "Wrong Q95");
}

void test_categorical_statistics() {
    const CategoricalColumn column{
        {"red", "blue", "red", std::nullopt}
    };

    const auto stats =
        calculate_categorical_statistics(column.values);

    check(
        stats.frequencies.at("red") == 2,
        "Wrong red frequency"
    );

    check(
        stats.frequencies.at("blue") == 1,
        "Wrong blue frequency"
    );
}

void test_preprocessing_combinations() {
    const auto dataset = make_example_dataset();
    const auto statistics = make_example_statistics(dataset);

    const std::vector<NumericImputation> imputations = {
        NumericImputation::Mean,
        NumericImputation::Median,
        NumericImputation::Zero
    };

    const std::vector<NumericScaling> scalings = {
        NumericScaling::MinMax,
        NumericScaling::Standard,
        NumericScaling::Robust
    };

    const std::vector<CategoryEncoding> encodings = {
        CategoryEncoding::OneHot,
        CategoryEncoding::Ordinal
    };

    std::size_t combinations = 0;

    for (const auto imputation : imputations) {
        for (const auto scaling : scalings) {
            for (const auto encoding : encodings) {
                const PreprocessingOptions options{
                    imputation,
                    scaling,
                    encoding
                };

                const auto matrix = preprocess_dataset(
                    dataset,
                    statistics,
                    options
                );

                const std::size_t expected_cols =
                    encoding == CategoryEncoding::OneHot
                        ? 4
                        : 2;

                check(matrix.rows == 4, "Wrong matrix rows");
                check(
                    matrix.cols == expected_cols,
                    "Wrong matrix columns"
                );

                check(
                    matrix.values.size() ==
                        matrix.rows * matrix.cols,
                    "Wrong matrix values count"
                );

                check(
                    matrix.target_values ==
                        std::vector<std::string>({
                            "YES", "NO", "YES", "NO"
                        }),
                    "Target values were changed"
                );

                // One-hot output:
                // age, color_blue, color_red, color___MISSING__
                if (encoding == CategoryEncoding::OneHot) {
                    check(
                        matrix.column_names[1] == "color_blue",
                        "Wrong one-hot column order"
                    );

                    check_close(
                        matrix.values[0 * matrix.cols + 2],
                        1.0,
                        "Wrong red one-hot value"
                    );

                    check_close(
                        matrix.values[1 * matrix.cols + 1],
                        1.0,
                        "Wrong blue one-hot value"
                    );

                    check_close(
                        matrix.values[2 * matrix.cols + 3],
                        1.0,
                        "Wrong missing one-hot value"
                    );
                } else {
                    // Alphabetical order: blue=0, red=1,
                    // missing=2.
                    check_close(
                        matrix.values[0 * matrix.cols + 1],
                        1.0,
                        "Wrong ordinal encoding for red"
                    );

                    check_close(
                        matrix.values[1 * matrix.cols + 1],
                        0.0,
                        "Wrong ordinal encoding for blue"
                    );

                    check_close(
                        matrix.values[2 * matrix.cols + 1],
                        2.0,
                        "Wrong ordinal encoding for missing"
                    );
                }

                // Source age values are 1, missing, 3, 5.
                // Statistics: mean=3, median=3,
                // min=1, max=5, variance=8/3,
                // Q25=2, Q75=4.
                const double fill =
                    imputation == NumericImputation::Zero
                        ? 0.0
                        : 3.0;

                double expected = 0.0;

                if (scaling == NumericScaling::MinMax) {
                    expected = (fill - 1.0) / 4.0;
                } else if (
                    scaling == NumericScaling::Standard
                ) {
                    expected =
                        (fill - 3.0) / std::sqrt(8.0 / 3.0);
                } else {
                    expected = (fill - 3.0) / 2.0;
                }

                check_close(
                    matrix.values[1 * matrix.cols],
                    expected,
                    "Wrong imputation or scaling result"
                );

                ++combinations;
            }
        }
    }

    check(
        combinations == 18,
        "Not all preprocessing combinations were checked"
    );
}

void test_binary_round_trip() {
    const auto dataset = make_example_dataset();
    const auto statistics = make_example_statistics(dataset);

    const PreprocessingOptions options{
        NumericImputation::Median,
        NumericScaling::MinMax,
        CategoryEncoding::OneHot
    };

    const auto original = preprocess_dataset(
        dataset,
        statistics,
        options
    );

    const std::string filename = "test_matrix.bin";

    save_binary_matrix(original, filename);
    const auto restored = load_binary_matrix(filename);

    check(restored.rows == original.rows, "Rows changed");
    check(restored.cols == original.cols, "Columns changed");

    check(
        restored.column_names == original.column_names,
        "Column names changed"
    );

    check(
        restored.values == original.values,
        "Matrix values changed"
    );

    check(
        restored.target_values == original.target_values,
        "Target values changed"
    );

    std::filesystem::remove(filename);
}

void test_invalid_binary() {
    const std::string filename = "test_bad_matrix.bin";

    write_text_file(filename, "INVALID");

    expect_exception(
        [&]() {
            (void)load_binary_matrix(filename);
        },
        "Invalid binary signature was accepted"
    );

    std::filesystem::remove(filename);
}

void test_truncated_binary() {
    const auto dataset = make_example_dataset();
    const auto statistics = make_example_statistics(dataset);

    const auto matrix = preprocess_dataset(
        dataset,
        statistics,
        PreprocessingOptions{}
    );

    const std::string filename = "test_truncated.bin";

    save_binary_matrix(matrix, filename);

    const auto original_size =
        std::filesystem::file_size(filename);

    std::filesystem::resize_file(
        filename,
        original_size - 1
    );

    expect_exception(
        [&]() {
            (void)load_binary_matrix(filename);
        },
        "Truncated binary file was accepted"
    );

    std::filesystem::remove(filename);
}

} // namespace

int main() {
    run_test("CSV loading and missing values", test_csv_loading);
    run_test("Invalid CSV", test_invalid_csv);
    run_test("Feature type detection", test_feature_type);
    run_test("Typed dataset", test_typed_dataset);
    run_test("Numeric statistics and quantiles", test_numeric_statistics);
    run_test("Categorical frequencies", test_categorical_statistics);
    run_test("All 18 preprocessing combinations",
             test_preprocessing_combinations);
    run_test("Binary round trip", test_binary_round_trip);
    run_test("Invalid binary signature", test_invalid_binary);
    run_test("Truncated binary file", test_truncated_binary);

    std::cout << "\nPassed: " << passed
              << "\nFailed: " << failed << '\n';

    return failed == 0 ? 0 : 1;
}