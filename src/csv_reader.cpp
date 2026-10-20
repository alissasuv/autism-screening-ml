#include "csv_reader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;

    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }

    if (!line.empty() && line.back() == ',') {
        fields.emplace_back();
    }

    return fields;
}

} // namespace

RawDataset read_csv(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + filename);
    }

    RawDataset dataset;
    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error("CSV file is empty");
    }

    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }

    dataset.column_names = split_csv_line(line);

    if (dataset.column_names.empty()) {
        throw std::runtime_error("CSV header is empty");
    }

    std::unordered_set<std::string> unique_names;

    for (const auto& name : dataset.column_names) {
        if (name.empty() || !unique_names.insert(name).second) {
            throw std::runtime_error(
                "CSV contains empty or duplicate column names"
            );
        }
    }

    dataset.columns.resize(dataset.column_names.size());

    std::size_t line_number = 1;

    while (std::getline(file, line)) {
        ++line_number;

        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        auto fields = split_csv_line(line);

        if (fields.size() != dataset.column_names.size()) {
            throw std::runtime_error(
                "Invalid number of fields at line " +
                std::to_string(line_number)
            );
        }

        for (std::size_t i = 0; i < fields.size(); ++i) {
            if (fields[i] == "?") {
                dataset.columns[i].push_back(std::nullopt);
            } else {
                dataset.columns[i].push_back(fields[i]);
            }
        }

        ++dataset.row_count;
    }

    if (file.bad()) {
        throw std::runtime_error("Error while reading CSV file");
    }

    return dataset;
}
