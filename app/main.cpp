#include "csv_reader.hpp"
#include "feature_type.hpp"
#include "statistics.hpp"

#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
    try {
        const auto dataset =
            read_csv("data/Autism-Adult-Data.csv");

        constexpr std::size_t histogram_bins = 10;

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
                const auto values =
                    extract_numeric_values(column);

                if (values.empty()) {
                    std::cout << "No numeric values.\n";
                    continue;
                }

                const auto stats =
                    calculate_numeric_statistics(values);

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
                    values, histogram_bins
                );
            } else {
                const auto stats =
                    calculate_categorical_statistics(column);

                print_categorical_histogram(stats);
            }
        }

    } catch (const std::exception& error) {
        std::cerr << "Error: "
                  << error.what() << '\n';
        return 1;
    }

    return 0;
}
