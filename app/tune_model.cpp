#include "lab5_data.hpp"
#include "lab5_preprocessing.hpp"
#include "search.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdlib>

namespace {

struct Arguments {
    std::string train;
    std::string target = "Class/ASD";
    std::string preprocessing_config;
    std::string search_config;
    std::string results_output;
    std::string best_config_output;
};

void usage() {
    std::cout
        << "Usage:\n"
        << "  tune_model"
        << " --train <path>"
        << " --preprocessing-config <yaml>"
        << " --config <yaml>"
        << " --results-output <csv>"
        << " --best-config-output <yaml>"
        << " [--target <column>]\n";
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments result;

    for (int i = 1; i < argc; ++i) {
        const std::string name = argv[i];

        if (name == "--help" || name == "-h") {
            usage();
            std::exit(0);
        }

        if (i + 1 >= argc) {
            throw std::invalid_argument(
                "Missing value for " + name
            );
        }

        const std::string value = argv[++i];

        if (name == "--train") {
            result.train = value;
        } else if (name == "--target") {
            result.target = value;
        } else if (name == "--preprocessing-config") {
            result.preprocessing_config = value;
        } else if (name == "--config") {
            result.search_config = value;
        } else if (name == "--results-output") {
            result.results_output = value;
        } else if (name == "--best-config-output") {
            result.best_config_output = value;
        } else {
            throw std::invalid_argument(
                "Unknown argument: " + name
            );
        }
    }

    if (result.train.empty() ||
        result.preprocessing_config.empty() ||
        result.search_config.empty() ||
        result.results_output.empty() ||
        result.best_config_output.empty()) {
        throw std::invalid_argument(
            "Missing required CLI arguments"
        );
    }

    return result;
}

void ensure_parent(const std::string& filename) {
    const auto parent =
        std::filesystem::path(filename).parent_path();

    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);

        const PreparedDataset prepared =
            load_prepared_dataset(args.train);

        const PreprocessingOptions preprocessing =
            lab5::load_preprocessing_config(
                args.preprocessing_config
            );

        const lab5::SearchConfig config =
            lab5::load_search_config(
                args.search_config
            );

        std::cout << "Lab 5 - Model Tuning\n";
        std::cout << "Task: "
                  << lab5::task_name(config.task) << '\n';

        std::cout << "Scoring: "
                  << lab5::scoring_name(config.scoring) << '\n';

        std::cout << "Training rows: "
                  << prepared.data.row_count << '\n';

        const lab5::SearchResult result =
            lab5::run_grid_search(
                prepared.data,
                args.target,
                preprocessing,
                config
            );

        ensure_parent(args.results_output);
        ensure_parent(args.best_config_output);

        lab5::save_search_results(
            result,
            config,
            args.results_output
        );

        lab5::save_best_model_config(
            result,
            config,
            args.best_config_output
        );

        const auto& best =
            result.records[result.best_index];

        std::cout
            << "\nBest candidate: "
            << (result.best_index + 1) << '\n'
            << "Best mean score: " << best.mean << '\n'
            << "CV stddev: " << best.stddev << '\n'
            << "Results: " << args.results_output << '\n'
            << "Best configuration: "
            << args.best_config_output << '\n';

        std::cout
            << "Model tuning completed successfully.\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "Error: " << error.what() << '\n';

        return 1;
    }
}


