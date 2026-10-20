#include "csv_reader.hpp"

#include <exception>
#include <iostream>

int main() {
    try {
        const auto dataset = read_csv("data/Autism-Adult-Data.csv");

        std::cout << "Dataset loaded successfully\n";
        std::cout << "Rows: " << dataset.row_count << '\n';
        std::cout << "Columns: " << dataset.column_names.size() << '\n';

        std::cout << "\nFeatures:\n";

        for (std::size_t i = 0; i < dataset.columns.size(); ++i) {
            std::size_t missing_count = 0;

            for (const auto& value : dataset.columns[i]) {
                if (!value.has_value()) {
                    ++missing_count;
                }
            }

            std::cout << i << ". "
                      << dataset.column_names[i]
                      << " (missing: " << missing_count << ")\n";
        }

    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
