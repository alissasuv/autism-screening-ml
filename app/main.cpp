#include "binary_io.hpp"
#include "cli.hpp"
#include "csv_reader.hpp"
#include "dataset.hpp"
#include "feature_type.hpp"
#include "preprocessing.hpp"
#include "statistics.hpp"

#include <cstddef>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

int main(int argc, char** argv) {
    try {
        const CliOptions config = parse_cli(argc, argv);

        if (config.show_help) {
            std::cout << config.help_text;
            return 0;
        }

        const auto raw_dataset = read_csv(config.input_path);

        std::cout << "Dataset loaded successfully\n";
        std::cout << "Rows: " << raw_dataset.row_count << '\n';
        std::cout << "Columns: "
                  << raw_dataset.column_names.size() << '\n';

        std::vector<FeatureType> types;
        types.reserve(raw_dataset.columns.size());

        for (const auto& column : raw_dataset.columns) {
            types.push_back(detect_feature_type(column));
        }

        // Convert raw strings into typed columns.
        const Dataset dataset =
            make_typed_dataset(raw_dataset, types);

        std::size_t total_missing = 0;
        std::size_t numerical_count = 0;
        std::size_t categorical_count = 0;

        std::cout << "\nFeatures:\n";

        for (std::size_t i = 0; i < dataset.columns.size(); ++i) {
            const auto& column = dataset.columns[i];

            const std::size_t missing = std::visit(
                [](const auto& typed_column) {
                    std::size_t count = 0;

                    for (const auto& cell : typed_column.values) {
                        if (!cell.has_value()) {
                            ++count;
                        }
                    }

                    return count;
                },
                column
            );

            total_missing += missing;

            const bool is_numeric =
                std::holds_alternative<NumericColumn>(column);

            if (is_numeric) {
                ++numerical_count;
            } else {
                ++categorical_count;
            }

            const double missing_percent =
                dataset.row_count == 0
                    ? 0.0
                    : 100.0 * missing / dataset.row_count;

            std::cout << i << ". "
                      << dataset.column_names[i]
                      << " ("
                      << (is_numeric ? "Numerical" : "Categorical")
                      << ", missing: " << missing
                      << ", " << std::fixed
                      << std::setprecision(2)
                      << missing_percent << "%)\n";
        }

        std::cout << "\nTotal missing values: "
                  << total_missing << '\n';

        std::cout << "\nTyped dataset created successfully.\n";
        std::cout << "Numerical columns: "
                  << numerical_count << '\n';
        std::cout << "Categorical columns: "
                  << categorical_count << '\n';

        // Calculate statistics before filling missing values.
        std::vector<std::optional<NumericStatistics>>
            numeric_statistics(dataset.columns.size());

        for (std::size_t i = 0; i < dataset.columns.size(); ++i) {
            const auto* numeric =
                std::get_if<NumericColumn>(&dataset.columns[i]);

            if (numeric == nullptr) {
                continue;
            }

            const auto values = extract_numeric_values(*numeric);

            if (!values.empty()) {
                numeric_statistics[i] =
                    calculate_numeric_statistics(values);
            }
        }

        // Interactive statistics and histograms.
        while (true) {
            std::cout << "\nEnter feature index (q to quit): ";

            std::string input;

            if (!std::getline(std::cin, input)) {
                break;
            }

            if (input == "q" || input == "Q") {
                break;
            }

            std::istringstream stream(input);
            std::size_t index = 0;
            std::string extra;

            if (!(stream >> index) ||
                (stream >> extra) ||
                index >= dataset.columns.size()) {
                std::cout << "Invalid feature index.\n";
                continue;
            }

            const auto& column = dataset.columns[index];

            const std::size_t missing = std::visit(
                [](const auto& typed_column) {
                    std::size_t count = 0;

                    for (const auto& cell : typed_column.values) {
                        if (!cell.has_value()) {
                            ++count;
                        }
                    }

                    return count;
                },
                column
            );

            std::cout << "\nFeature: "
                      << dataset.column_names[index]
                      << "\nMissing values: "
                      << missing << '\n';

            if (const auto* numeric =
                    std::get_if<NumericColumn>(&column)) {

                if (!numeric_statistics[index].has_value()) {
                    std::cout
                        << "No numeric statistics available.\n";
                    continue;
                }

                const auto& stats = *numeric_statistics[index];
                const auto values = extract_numeric_values(*numeric);

                std::cout << std::fixed
                          << std::setprecision(3);

                std::cout << "Min: " << stats.min << '\n';
                std::cout << "Max: " << stats.max << '\n';
                std::cout << "Mean: " << stats.mean << '\n';
                std::cout << "Variance: "
                          << stats.variance << '\n';

                std::cout << "Q05: " << stats.q05 << '\n';
                std::cout << "Q25: " << stats.q25 << '\n';
                std::cout << "Q50: " << stats.q50 << '\n';
                std::cout << "Q75: " << stats.q75 << '\n';
                std::cout << "Q95: " << stats.q95 << '\n';

                print_numeric_histogram(
                    values,
                    config.histogram_bins
                );

            } else {
                const auto& categorical =
                    std::get<CategoricalColumn>(column);

                const auto stats =
                    calculate_categorical_statistics(
                        categorical.values
                    );

                print_categorical_histogram(stats);
            }
        }

        std::cout << "\nPreprocess dataset? (y/N): ";

        std::string answer;

        if (std::getline(std::cin, answer) &&
            (answer == "y" || answer == "Y")) {

            const auto matrix = preprocess_dataset(
                dataset,
                numeric_statistics,
                config.preprocessing
            );

            std::cout << "Preprocessing completed.\n";
            std::cout << "Matrix rows: "
                      << matrix.rows << '\n';
            std::cout << "Matrix columns: "
                      << matrix.cols << '\n';
            std::cout << "Matrix values: "
                      << matrix.values.size() << '\n';
            std::cout << "Target values: "
                      << matrix.target_values.size() << '\n';

            std::cout << "\nOutput columns:\n";

            for (std::size_t i = 0; i < matrix.cols; ++i) {
                std::cout << i << ". "
                          << matrix.column_names[i] << '\n';
            }

            // Save processed data.
            save_binary_matrix(matrix, config.output_path);

            // Read the file and verify all data.
            const auto restored =
                load_binary_matrix(config.output_path);

            if (restored.rows != matrix.rows ||
                restored.cols != matrix.cols ||
                restored.column_names != matrix.column_names ||
                restored.target_values != matrix.target_values ||
                restored.values != matrix.values) {

                throw std::runtime_error(
                    "Binary round-trip verification failed"
                );
            }

            const auto csv_size =
                std::filesystem::file_size(config.input_path);

            const auto binary_size =
                std::filesystem::file_size(config.output_path);

            std::cout << "\nBinary file saved: "
                      << config.output_path << '\n';

            std::cout
                << "Binary round-trip verification passed.\n";

            std::cout << "\nFile sizes:\n";
            std::cout << "CSV: "
                      << csv_size << " bytes\n";
            std::cout << "Binary: "
                      << binary_size << " bytes\n";
        }

    } catch (const std::exception& error) {
        std::cerr << "Error: "
                  << error.what() << '\n';
        return 1;
    }

    return 0;
}