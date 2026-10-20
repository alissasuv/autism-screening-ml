#include "binary_io.hpp"
#include "cli.hpp"
#include "csv_reader.hpp"
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
#include <vector>

int main(int argc, char** argv) {
    try {
        const CliOptions config = parse_cli(argc, argv);

        if (config.show_help) {
            std::cout << config.help_text;
            return 0;
        }

        const auto dataset = read_csv(config.input_path);

        std::cout << "Dataset loaded successfully\n";
        std::cout << "Rows: " << dataset.row_count << '\n';
        std::cout << "Columns: "
                  << dataset.column_names.size() << '\n';

        std::vector<FeatureType> types;
        types.reserve(dataset.columns.size());

        std::size_t total_missing = 0;

        std::cout << "\nFeatures:\n";

        for (std::size_t i = 0;
             i < dataset.columns.size(); ++i) {

            const auto& column = dataset.columns[i];
            const auto type = detect_feature_type(column);
            types.push_back(type);

            std::size_t missing = 0;

            for (const auto& cell : column) {
                if (!cell.has_value()) {
                    ++missing;
                }
            }

            total_missing += missing;

            const double missing_percent =
                dataset.row_count == 0 ? 0.0 :
                100.0 * missing / dataset.row_count;

            std::cout << i << ". "
                      << dataset.column_names[i]
                      << " ("
                      << (type == FeatureType::Numerical
                          ? "Numerical" : "Categorical")
                      << ", missing: " << missing
                      << ", " << std::fixed
                      << std::setprecision(2)
                      << missing_percent << "%)\n";
        }

        std::cout << "\nTotal missing values: "
                  << total_missing << '\n';

        // Calculate numeric statistics before preprocessing.
        std::vector<std::optional<NumericStatistics>>
            numeric_statistics(dataset.columns.size());

        for (std::size_t i = 0;
             i < dataset.columns.size(); ++i) {

            if (types[i] != FeatureType::Numerical) {
                continue;
            }

            const auto values =
                extract_numeric_values(dataset.columns[i]);

            if (!values.empty()) {
                numeric_statistics[i] =
                    calculate_numeric_statistics(values);
            }
        }

        // Interactive feature analysis.
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
            std::size_t index;
            std::string extra;

            if (!(stream >> index) ||
                (stream >> extra) ||
                index >= dataset.columns.size()) {
                std::cout << "Invalid feature index.\n";
                continue;
            }

            const auto& column = dataset.columns[index];

            std::size_t missing = 0;

            for (const auto& cell : column) {
                if (!cell.has_value()) {
                    ++missing;
                }
            }

            std::cout << "\nFeature: "
                      << dataset.column_names[index]
                      << "\nMissing values: "
                      << missing << '\n';

            if (types[index] == FeatureType::Numerical) {
                if (!numeric_statistics[index].has_value()) {
                    std::cout
                        << "No numeric statistics available.\n";
                    continue;
                }

                const auto& stats = *numeric_statistics[index];
                const auto values =
                    extract_numeric_values(column);

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
                const auto stats =
                    calculate_categorical_statistics(column);

                print_categorical_histogram(stats);
            }
        }

        // Optional preprocessing and binary saving.
        std::cout << "\nPreprocess dataset? (y/N): ";

        std::string answer;

        if (std::getline(std::cin, answer) &&
            (answer == "y" || answer == "Y")) {

            const auto matrix = preprocess_dataset(
                dataset,
                types,
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

            // Save the processed matrix to a binary file.
            save_binary_matrix(matrix, config.output_path);

            // Load it again to verify the saved data.
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