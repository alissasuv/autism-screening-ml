#include "csv_reader.hpp"
#include "evaluation.hpp"
#include "feature_importance.hpp"
#include "lab5_preprocessing.hpp"
#include "search.hpp"
#include "validation.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct RegressionSplit {
    RawDataset train;
    RawDataset test;
    std::vector<std::uint64_t> test_indices;
};

RegressionSplit make_split(const RawDataset& full) {
    std::vector<std::size_t> train_indices;
    std::vector<std::size_t> test_indices;

    for (std::size_t i = 0; i < full.row_count; ++i) {
        if (i % 4 == 0) {
            test_indices.push_back(i);
        } else {
            train_indices.push_back(i);
        }
    }

    require(
        !train_indices.empty() && !test_indices.empty(),
        "Train and test must be nonempty"
    );

    RegressionSplit result;

    result.train = lab5::select_raw_rows(
        full, train_indices
    );

    result.test = lab5::select_raw_rows(
        full, test_indices
    );

    for (const auto index : test_indices) {
        result.test_indices.push_back(
            static_cast<std::uint64_t>(index)
        );
    }

    return result;
}

void run_family(
    const std::string& family,
    const std::string& config_path,
    const RegressionSplit& split,
    const PreprocessingOptions& preprocessing,
    const std::filesystem::path& results_dir
) {
    const lab5::SearchConfig config =
        lab5::load_search_config(config_path);

    require(
        config.task == lab5::Task::Regression,
        "Smoke test must use regression"
    );

    require(
        config.scoring == lab5::Scoring::MSE,
        "Smoke test must select models using MSE"
    );

    require(
        config.n_splits == 2,
        "Smoke test requires two CV folds"
    );

    const auto search = lab5::run_grid_search(
        split.train,
        "target",
        preprocessing,
        config
    );

    require(
        search.has_best,
        "No valid regression model was found"
    );

    const auto expected_candidates =
        family == "deep" ? 2U : 4U;

    require(
        search.records.size() == expected_candidates,
        "Unexpected search grid size"
    );

    const auto prefix = results_dir /
        ("smoke_regression_" + family);

    const std::string search_path =
        prefix.string() + "_search.csv";

    const std::string config_output =
        prefix.string() + "_best.yaml";

    const std::string metrics_path =
        prefix.string() + "_metrics.csv";

    const std::string predictions_path =
        prefix.string() + "_predictions.csv";

    const std::string importance_path =
        prefix.string() + "_importance.csv";

    lab5::save_search_results(
        search, config, search_path
    );

    lab5::save_best_model_config(
        search, config, config_output
    );

    // Re-read the saved configuration: the smoke test
    // must verify the actual serialized best parameters.
    const YAML::Node saved =
        YAML::LoadFile(config_output);

    require(
        saved["task"].as<std::string>() == "regression",
        "Saved task is incorrect"
    );

    const auto scoring = lab5::parse_scoring(
        saved["scoring"].as<std::string>()
    );

    const auto model = lab5::parse_model_config(
        lab5::Task::Regression,
        saved["model"]["params"]
    );

    // Search uses CV train folds; final training
    // uses the complete training partition.
    const auto trained = lab5::train_final_model(
        split.train,
        "target",
        preprocessing,
        model,
        scoring
    );

    const auto evaluation = lab5::evaluate_final_model(
        trained,
        split.train,
        split.test
    );

    require(
        evaluation.metrics.size() == 3,
        "Expected MSE, MAE and R2"
    );

    require(
        evaluation.metrics[0].name == "mse" &&
        evaluation.metrics[1].name == "mae" &&
        evaluation.metrics[2].name == "r2",
        "Unexpected regression metric names"
    );

    for (const auto& metric : evaluation.metrics) {
        require(
            std::isfinite(metric.train) &&
            std::isfinite(metric.test),
            "Regression metric is not finite"
        );
    }

    require(
        evaluation.test_predictions.predicted.size() ==
            split.test.row_count,
        "Incorrect prediction count"
    );

    lab5::save_evaluation_metrics(
        evaluation,
        metrics_path
    );

    lab5::save_evaluation_predictions(
        evaluation,
        split.test_indices,
        predictions_path
    );

    const auto importance =
        lab5::permutation_feature_importance(
            trained,
            split.test,
            3,
            123
        );

    require(
        importance.features.size() == 2,
        "Expected exactly two original features"
    );

    require(
        std::isfinite(importance.baseline_score),
        "Baseline is not finite"
    );

    for (const auto& feature : importance.features) {
        require(
            feature.repeat_importances.size() == 3,
            "Incorrect importance repeat count"
        );

        require(
            std::isfinite(feature.mean_importance) &&
            std::isfinite(feature.stddev),
            "Invalid permutation importance"
        );

        require(
            feature.feature_name != "target",
            "Target was included in feature importance"
        );
    }

    lab5::save_feature_importance(
        importance,
        importance_path
    );

    // Prediction and scoring after importance must not
    // change: the fitted objects are reused as-is.
    const double score_again =
        lab5::score_final_model(trained, split.test);

    require(
        std::abs(score_again - importance.baseline_score)
            < 1e-10,
        "Permutation importance changed the trained model"
    );

    require(
        std::filesystem::exists(search_path) &&
        std::filesystem::exists(config_output) &&
        std::filesystem::exists(metrics_path) &&
        std::filesystem::exists(predictions_path) &&
        std::filesystem::exists(importance_path),
        "Some regression outputs were not saved"
    );

    std::cout
        << "\n" << family << " regression forest:\n"
        << "Best CV MSE: "
        << search.records[search.best_index].mean
        << '\n';

    for (const auto& metric : evaluation.metrics) {
        std::cout
            << metric.name << ": train="
            << metric.train
            << ", test="
            << metric.test
            << '\n';
    }

    std::cout
        << "Baseline permutation score: "
        << importance.baseline_score << '\n'
        << "[PASS] " << family << " regression forest\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(
            argc == 5,
            "Usage: test_lab5_regression_smoke "
            "<csv> <preprocessing.yaml> "
            "<deep.yaml> <shallow.yaml>"
        );

        const RawDataset full = read_csv(argv[1]);

        require(
            full.row_count == 24,
            "Expected 24 synthetic regression rows"
        );

        const RegressionSplit split = make_split(full);

        require(
            split.train.row_count == 18 &&
            split.test.row_count == 6,
            "Unexpected regression train/test sizes"
        );

        const PreprocessingOptions preprocessing =
            lab5::load_preprocessing_config(argv[2]);

        const auto results_dir =
            std::filesystem::path("results");

        std::filesystem::create_directories(results_dir);

        run_family(
            "deep",
            argv[3],
            split,
            preprocessing,
            results_dir
        );

        run_family(
            "shallow",
            argv[4],
            split,
            preprocessing,
            results_dir
        );

        std::cout
            << "\nAll Lab 5 regression smoke checks passed.\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "[FAIL] " << error.what() << '\n';

        return 1;
    }
}