#include "search.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

RawDataset make_data() {
    RawDataset data;

    data.column_names = {"x", "category", "target"};
    data.row_count = 12;

    data.columns.resize(3);

    for (std::size_t i = 0; i < 12; ++i) {
        data.columns[0].push_back(std::to_string(i));

        data.columns[1].push_back(
            i % 2 == 0 ? "red" : "blue"
        );

        data.columns[2].push_back(
            i < 6 ? "NO" : "YES"
        );
    }

    return data;
}

void test_yaml_and_grid(
    const std::string& filename
) {
    auto config = lab5::load_search_config(filename);

    require(
        config.task == lab5::Task::Classification,
        "Incorrect task"
    );

    require(
        config.scoring == lab5::Scoring::F1,
        "Incorrect scoring"
    );

    require(config.n_splits == 3, "Incorrect fold count");

    // Use a tiny ensemble for the unit test.
    config.base_params["n_estimators"] = 2;
    config.base_params["n_jobs"] = 1;

    for (auto& axis : config.axes) {
        if (axis.name == "n_estimators") {
            axis.values.clear();
            axis.values.push_back(YAML::Node(2));
            axis.values.push_back(YAML::Node(3));
        }
    }

    const PreprocessingOptions preprocessing{};

    const auto first = lab5::run_grid_search(
        make_data(),
        "target",
        preprocessing,
        config
    );

    const auto second = lab5::run_grid_search(
        make_data(),
        "target",
        preprocessing,
        config
    );

    require(first.has_best, "No valid best model");

    require(
        first.records.size() == 4,
        "Incorrect deep forest grid size"
    );

    require(
        first.best_index == second.best_index,
        "Best candidate is not reproducible"
    );

    for (std::size_t i = 0; i < first.records.size(); ++i) {
        require(
            first.records[i].valid,
            "Unexpected invalid forest candidate"
        );

        require(
            first.records[i].fold_scores ==
                second.records[i].fold_scores,
            "CV scores are not reproducible"
        );
    }

    const auto directory =
        std::filesystem::temp_directory_path();

    const auto csv = directory / "lab5_search_test.csv";
    const auto yaml = directory / "lab5_best_test.yaml";

    lab5::save_search_results(
        first, config, csv.string()
    );

    lab5::save_best_model_config(
        first, config, yaml.string()
    );

    require(
        std::filesystem::exists(csv),
        "Search CSV was not saved"
    );

    const YAML::Node saved = YAML::LoadFile(
        yaml.string()
    );

    require(
        saved["model"]["type"].as<std::string>()
            == "random_forest",
        "Saved model type is incorrect"
    );

    (void)lab5::parse_model_config(
        lab5::Task::Classification,
        saved["model"]["params"]
    );

    std::filesystem::remove(csv);
    std::filesystem::remove(yaml);

    std::cout << "[PASS] YAML grid search and persistence\n";
}

void test_metric_direction() {
    require(
        lab5::is_better_score(
            0.8, 0.7, lab5::Scoring::F1
        ),
        "Higher F1 should win"
    );

    require(
        lab5::is_better_score(
            0.1, 0.2, lab5::Scoring::MSE
        ),
        "Lower MSE should win"
    );

    require(
        !lab5::is_better_score(
            0.7, 0.7, lab5::Scoring::F1
        ),
        "Equal scores must keep the first candidate"
    );

    std::cout << "[PASS] Metric direction\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Expected YAML config path");

        test_metric_direction();
        test_yaml_and_grid(argv[1]);

        std::cout
            << "All Lab 5 search tests passed.\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}







