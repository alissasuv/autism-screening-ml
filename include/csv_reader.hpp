
#pragma once

#include <optional>
#include <string>
#include <vector>

struct CsvConfig {
    char delimiter = ',';
    std::string missing_value = "?";
    bool has_header = true;
};

struct RawDataset {
    std::vector<std::string> column_names;

    std::vector<std::vector<std::optional<std::string>>> columns;

    std::size_t row_count = 0;
};

RawDataset read_csv(
    const std::string& filename,
    const CsvConfig& config = CsvConfig{}
);
