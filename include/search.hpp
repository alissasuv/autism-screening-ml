#pragma once

#include "csv_reader.hpp"
#include "metrics.hpp"
#include "model_factory.hpp"
#include "preprocessing.hpp"

#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lab5 {

struct SearchAxis {
    std::string name;
    std::vector<YAML::Node> values;
};

struct SearchConfig {
    Task task = Task::Classification;
    Scoring scoring = Scoring::F1;

    YAML::Node base_params;

    std::size_t n_splits = 3;
    std::uint32_t cv_seed = 17;

    std::vector<SearchAxis> axes;
};

struct SearchRecord {
    YAML::Node params;

    std::vector<double> fold_scores;
    double mean = 0.0;
    double stddev = 0.0;

    bool valid = false;
    std::string error;
};

struct SearchResult {
    std::vector<SearchRecord> records;
    std::size_t best_index = 0;
    bool has_best = false;
};

SearchConfig load_search_config(
    const std::string& filename
);

ModelConfig parse_model_config(
    Task task,
    const YAML::Node& params
);

SearchResult run_grid_search(
    const RawDataset& train,
    const std::string& target_name,
    const PreprocessingOptions& preprocessing,
    const SearchConfig& config
);

void save_search_results(
    const SearchResult& result,
    const SearchConfig& config,
    const std::string& filename
);

void save_best_model_config(
    const SearchResult& result,
    const SearchConfig& config,
    const std::string& filename
);

std::string task_name(Task task);

} // namespace lab5





