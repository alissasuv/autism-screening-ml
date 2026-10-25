#include "csv_reader.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

std::vector<std::string> split_csv_line(
    const std::string& line,
    char delimiter
) {
    std::vector<std::string> fields;
    std::string current;

    for (char symbol : line) {
        if (symbol == delimiter) {
            fields.push_back(std::move(current));
            current.clear();
        } else {
            current += symbol;
        }
    }

    fields.push_back(std::move(current));

    return fields;
}

} // namespace

RawDataset read_csv(
    const std::string& filename,
    const CsvConfig& config
) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot open CSV file: " + filename
        );
    }

    if (config.delimiter == '\n' ||
        config.delimiter == '\r') {
        throw std::runtime_error(
            "Invalid CSV delimiter"
        );
    }

    RawDataset dataset;

    std::string line;
    std::size_t line_number = 0;

    if (!std::getline(file, line)) {
        throw std::runtime_error(
            "CSV file is empty"
        );
    }

    ++line_number;

    // Handle Windows-style line endings.
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }

    auto first_fields = split_csv_line(
        line, config.delimiter
    );

    if (config.has_header) {
        dataset.column_names = std::move(first_fields);
    } else {
        for (std::size_t i = 0;
             i < first_fields.size(); ++i) {
            dataset.column_names.push_back(
                "column_" + std::to_string(i)
            );
        }
    }

    if (dataset.column_names.empty()) {
        throw std::runtime_error(
            "CSV contains no columns"
        );
    }

    std::unordered_set<std::string> unique_names;

    for (const auto& name : dataset.column_names) {
        if (name.empty()) {
            throw std::runtime_error(
                "CSV contains an empty column name"
            );
        }

        if (!unique_names.insert(name).second) {
            throw std::runtime_error(
                "Duplicate CSV column name: " + name
            );
        }
    }

    dataset.columns.resize(dataset.column_names.size());

    // Add the first row when no header is present.
    if (!config.has_header) {
        for (std::size_t i = 0;
             i < first_fields.size(); ++i) {
            if (first_fields[i] == config.missing_value ||
                first_fields[i].empty()) {
                dataset.columns[i].push_back(std::nullopt);
            } else {
                dataset.columns[i].push_back(first_fields[i]);
            }
        }

        ++dataset.row_count;
    }

    while (std::getline(file, line)) {
        ++line_number;

        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        auto fields = split_csv_line(
            line, config.delimiter
        );

        if (fields.size() != dataset.columns.size()) {
            throw std::runtime_error(
                "Invalid number of fields at line " +
                std::to_string(line_number) +
                ": expected " +
                std::to_string(dataset.columns.size()) +
                ", got " +
                std::to_string(fields.size())
            );
        }

        for (std::size_t i = 0; i < fields.size(); ++i) {
            if (fields[i] == config.missing_value ||
                fields[i].empty()) {
                dataset.columns[i].push_back(std::nullopt);
            } else {
                dataset.columns[i].push_back(
                    std::move(fields[i])
                );
            }
        }

        ++dataset.row_count;
    }

    if (file.bad()) {
        throw std::runtime_error(
            "Error while reading CSV file"
        );
    }

    if (dataset.row_count == 0) {
        throw std::runtime_error(
            "CSV contains no data rows"
        );
    }

    return dataset;
}