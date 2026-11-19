#include "feature_importance.hpp"
#include "model_factory.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(
    bool condition,
    const std::string& message
) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool close(double a, double b) {
    return std::abs(a - b) < 1e-10;
}

RawDataset make_data(lab5::Task task) {
    RawDataset data;
    data.row_count = 20;
    data.column_names = {"number", "color", "target"};
    data.columns.resize(3);

    for (std::size_t i = 0; i < data.row_count; ++i) {
        data.columns[0].push_back(std::to_string(i));

        // Include missing categorical values.
        if (i % 5 == 0) {
            data.columns[1].push_back(std::nullopt);
        } else {
            data.columns[1].push_back(
                i % 2 == 0 ? "red" : "blue"
            );
        }

        if (task == lab5::Task::Classification) {
            data.columns[2].push_back(
                i < 10 ? "NO" : "YES"
            );
        } else {
            data.columns[2].push_back(
                std::to_string(i * 3)
            );
        }
    }

    return data;
}

lab5::ModelConfig make_model(lab5::Task task) {
    lab5::ModelConfig config;
    config.task = task;

    config.parameters.n_estimators = 3;
    config.parameters.n_jobs = 1;
    config.parameters.random_state = 42;
    config.parameters.bootstrap = false;
    config.parameters.max_features = {
        FeatureMode::All, 1
    };
    config.parameters.tree = {4, 2, 1};

    return config;
}

void test_importance(lab5::Task task) {
    const RawDataset data = make_data(task);
    const RawDataset original = data;

    const lab5::Scoring scoring =
        task == lab5::Task::Classification
            ? lab5::Scoring::F1
            : lab5::Scoring::MAE;

    const auto trained = lab5::train_final_model(
        data,
        "target",
        PreprocessingOptions{},
        make_model(task),
        scoring
    );

    const double baseline =
        lab5::score_final_model(trained, data);

    const auto first =
        lab5::permutation_feature_importance(
            trained, data, 5, 123
        );

    const auto second =
        lab5::permutation_feature_importance(
            trained, data, 5, 123
        );

    require(
        close(first.baseline_score, baseline),
        "Incorrect baseline"
    );

    require(
        close(first.baseline_score, second.baseline_score),
        "Baseline changed"
    );

    require(
        first.features.size() == 2,
        "Target was included in feature importance"
    );

    require(
        data.row_count == original.row_count &&
        data.column_names == original.column_names &&
        data.columns == original.columns,
        "Original raw data was modified"
    );

    require(
        first.features.size() == second.features.size(),
        "Importance size changed"
    );

    for (std::size_t i = 0;
         i < first.features.size();
         ++i) {
        const auto& left = first.features[i];
        const auto& right = second.features[i];

        require(
            left.feature_name == right.feature_name &&
            left.feature_index == right.feature_index,
            "Feature order is not reproducible"
        );

        require(
            left.repeat_importances.size() == 5,
            "Incorrect repeat count"
        );

        require(
            left.repeat_importances ==
                right.repeat_importances,
            "Permutation scores are not reproducible"
        );

        require(
            close(left.mean_importance,
                  right.mean_importance) &&
            close(left.stddev, right.stddev),
            "Importance summary changed"
        );

        require(
            std::isfinite(left.mean_importance) &&
            std::isfinite(left.stddev) &&
            left.stddev >= 0.0,
            "Invalid importance summary"
        );
    }

    require(
        first.features[0].mean_importance >=
            first.features[1].mean_importance,
        "Features are not sorted"
    );

    // Scoring the original data again must not require
    // retraining or alter the fitted model.
    require(
        close(
            lab5::score_final_model(trained, data),
            baseline
        ),
        "Trained model changed during permutations"
    );

    const auto filename =
        std::filesystem::temp_directory_path() /
        (task == lab5::Task::Classification
            ? "lab5_importance_classification.csv"
            : "lab5_importance_regression.csv");

    lab5::save_feature_importance(
        first, filename.string()
    );

    require(
        std::filesystem::exists(filename),
        "Importance CSV was not created"
    );

    std::filesystem::remove(filename);

    std::cout
        << "[PASS] Permutation importance: "
        << (task == lab5::Task::Classification
            ? "classification" : "regression")
        << '\n';
}

void test_invalid_repeats() {
    const RawDataset data = make_data(
        lab5::Task::Classification
    );

    const auto trained = lab5::train_final_model(
        data,
        "target",
        PreprocessingOptions{},
        make_model(lab5::Task::Classification),
        lab5::Scoring::F1
    );

    bool caught = false;

    try {
        (void)lab5::permutation_feature_importance(
            trained, data, 0, 123
        );
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    require(
        caught,
        "Zero permutation repeats were accepted"
    );

    std::cout << "[PASS] Invalid repeat count\n";
}

} // namespace

int main() {
    try {
        test_importance(lab5::Task::Classification);
        test_importance(lab5::Task::Regression);
        test_invalid_repeats();

        std::cout
            << "All Lab 5 importance tests passed.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "[FAIL] " << error.what() << '\n';

        return 1;
    }
}

