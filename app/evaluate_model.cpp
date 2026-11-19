
#include "evaluation.hpp"
#include "feature_importance.hpp"
#include "lab5_data.hpp"
#include "lab5_preprocessing.hpp"
#include "search.hpp"

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

struct Arguments {
    std::string train;
    std::string test;
    std::string target = "Class/ASD";

    std::string preprocessing_config;
    std::string model_config;

    std::string metrics_output;
    std::string predictions_output;

    bool feature_importance = false;
    std::size_t importance_repeats = 5;
    std::uint32_t importance_random_state = 123;
    std::string importance_output;
};

void usage() {
    std::cout
        << "Usage:\n"
        << "  evaluate_model"
        << " --train <file>"
        << " --test <file>"
        << " --preprocessing-config <yaml>"
        << " --model-config <yaml>"
        << " --metrics-output <csv>"
        << " --predictions-output <csv>"
        << " [--target <column>]"
        << " [--feature-importance"
        << " --importance-repeats <positive_integer>"
        << " --importance-random-state <seed>"
        << " --importance-output <csv>]\n";
}

std::uint64_t parse_unsigned(
    const std::string& text,
    const std::string& argument
) {
    if (text.empty() ||
        text[0] == '-' ||
        text[0] == '+') {
        throw std::invalid_argument(
            "Invalid unsigned value for " + argument
        );
    }

    std::size_t position = 0;
    const auto result = std::stoull(text, &position);

    if (position != text.size()) {
        throw std::invalid_argument(
            "Invalid unsigned value for " + argument
        );
    }

    return result;
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];

        if (key == "--help" || key == "-h") {
            usage();
            std::exit(0);
        }

        if (key == "--feature-importance") {
            args.feature_importance = true;
            continue;
        }

        if (i + 1 >= argc) {
            throw std::invalid_argument(
                "Missing value for argument: " + key
            );
        }

        const std::string value = argv[++i];

        if (key == "--train") {
            args.train = value;
        } else if (key == "--test") {
            args.test = value;
        } else if (key == "--target") {
            args.target = value;
        } else if (key == "--preprocessing-config") {
            args.preprocessing_config = value;
        } else if (key == "--model-config") {
            args.model_config = value;
        } else if (key == "--metrics-output") {
            args.metrics_output = value;
        } else if (key == "--predictions-output") {
            args.predictions_output = value;
        } else if (key == "--importance-repeats") {
            const auto parsed = parse_unsigned(value, key);

            if (parsed == 0 ||
                parsed >
                    std::numeric_limits<std::size_t>::max()) {
                throw std::invalid_argument(
                    "Invalid importance repeat count"
                );
            }

            args.importance_repeats =
                static_cast<std::size_t>(parsed);
        } else if (key == "--importance-random-state") {
            const auto parsed = parse_unsigned(value, key);

            if (parsed >
                std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument(
                    "Importance seed is out of range"
                );
            }

            args.importance_random_state =
                static_cast<std::uint32_t>(parsed);
        } else if (key == "--importance-output") {
            args.importance_output = value;
        } else {
            throw std::invalid_argument(
                "Unknown argument: " + key
            );
        }
    }

    if (args.train.empty() ||
        args.test.empty() ||
        args.preprocessing_config.empty() ||
        args.model_config.empty() ||
        args.metrics_output.empty() ||
        args.predictions_output.empty()) {
        throw std::invalid_argument(
            "Missing required CLI arguments"
        );
    }

    if (args.feature_importance &&
        args.importance_output.empty()) {
        throw std::invalid_argument(
            "--importance-output is required with "
            "--feature-importance"
        );
    }

    return args;
}

void ensure_parent(const std::string& filename) {
    const auto parent =
        std::filesystem::path(filename).parent_path();

    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

struct LoadedModelConfig {
    lab5::ModelConfig model;
    lab5::Scoring scoring;
};

LoadedModelConfig load_model_configuration(
    const std::string& filename
) {
    const YAML::Node root = YAML::LoadFile(filename);

    if (!root.IsMap()) {
        throw std::invalid_argument(
            "Model configuration must be a YAML mapping"
        );
    }

    const std::string task_name =
        root["task"].as<std::string>();

    lab5::Task task;

    if (task_name == "classification") {
        task = lab5::Task::Classification;
    } else if (task_name == "regression") {
        task = lab5::Task::Regression;
    } else {
        throw std::invalid_argument(
            "Unknown model task: " + task_name
        );
    }

    const auto scoring = lab5::parse_scoring(
        root["scoring"].as<std::string>()
    );

    lab5::validate_scoring(
        scoring,
        task == lab5::Task::Classification
    );

    const YAML::Node model = root["model"];

    if (!model.IsMap() ||
        model["type"].as<std::string>() != "random_forest") {
        throw std::invalid_argument(
            "Expected random_forest model"
        );
    }

    if (!model["params"].IsMap()) {
        throw std::invalid_argument(
            "Missing model.params mapping"
        );
    }

    return {
        lab5::parse_model_config(task, model["params"]),
        scoring
    };
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);

        const PreparedDataset train =
            load_prepared_dataset(args.train);

        const PreparedDataset test =
            load_prepared_dataset(args.test);

        const PreprocessingOptions preprocessing =
            lab5::load_preprocessing_config(
                args.preprocessing_config
            );

        const LoadedModelConfig config =
            load_model_configuration(
                args.model_config
            );

        std::cout << "Lab 5 - Final Evaluation\n";
        std::cout << "Task: "
                  << lab5::task_name(config.model.task)
                  << '\n';

        std::cout << "Scoring: "
                  << lab5::scoring_name(config.scoring)
                  << '\n';

        std::cout << "Train rows: "
                  << train.data.row_count << '\n';

        std::cout << "Test rows: "
                  << test.data.row_count << '\n';

        // Fit the model and preprocessing only once.
        const lab5::TrainedEvaluation trained =
            lab5::train_final_model(
                train.data,
                args.target,
                preprocessing,
                config.model,
                config.scoring
            );

        const lab5::FinalEvaluation evaluation =
            lab5::evaluate_final_model(
                trained,
                train.data,
                test.data
            );

        ensure_parent(args.metrics_output);
        ensure_parent(args.predictions_output);

        lab5::save_evaluation_metrics(
            evaluation,
            args.metrics_output
        );

        lab5::save_evaluation_predictions(
            evaluation,
            test.source_indices,
            args.predictions_output
        );

        std::cout << "\nMetrics (train / test)\n";

        for (const auto& metric : evaluation.metrics) {
            std::cout
                << metric.name << ": "
                << metric.train << " / "
                << metric.test << '\n';
        }

        std::cout
            << "\nMetrics: " << args.metrics_output
            << '\n'
            << "Predictions: " << args.predictions_output
            << '\n';

        if (args.feature_importance) {
            std::cout
                << "\nCalculating permutation "
                << "feature importance...\n"
                << "Repeats: "
                << args.importance_repeats << '\n'
                << "Random state: "
                << args.importance_random_state << '\n';

            const auto importance =
                lab5::permutation_feature_importance(
                    trained,
                    test.data,
                    args.importance_repeats,
                    args.importance_random_state
                );

            ensure_parent(args.importance_output);

            lab5::save_feature_importance(
                importance,
                args.importance_output
            );

            std::cout
                << "Baseline score: "
                << importance.baseline_score << '\n';

            std::cout << "\nFeature importance:\n";

            for (const auto& feature :
                 importance.features) {
                std::cout
                    << feature.feature_name
                    << ": mean="
                    << feature.mean_importance
                    << ", std="
                    << feature.stddev
                    << '\n';
            }

            std::cout
                << "\nImportance: "
                << args.importance_output
                << '\n';
        }

        std::cout
            << "Evaluation completed successfully.\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "Error: " << error.what()
            << '\n';

        return 1;
    }
}
