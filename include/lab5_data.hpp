#pragma once

#include "csv_reader.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct PreparedDataset {
    RawDataset data;
    std::vector<std::uint64_t> source_indices;
};

struct PreparedSplit {
    PreparedDataset train;
    PreparedDataset test;
};

PreparedSplit make_stratified_split(
    const RawDataset& source,
    const std::string& target_name,
    double test_fraction,
    std::uint32_t seed
);

void save_prepared_dataset(
    const PreparedDataset& dataset,
    const std::string& filename
);

PreparedDataset load_prepared_dataset(
    const std::string& filename
);

void save_dataset_statistics(
    const RawDataset& source,
    const PreparedSplit& split,
    const std::string& target_name,
    double test_fraction,
    std::uint32_t seed,
    const std::string& filename
);



