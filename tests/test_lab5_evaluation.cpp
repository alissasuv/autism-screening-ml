#include "evaluation.hpp"
#include "model_factory.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
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

RawDataset make_dataset(lab5::Task task) {
    RawDataset data;
    data.row_count = 12;
    data.column_names = {"x", "color", "target"};
    data.columns.resize(3);

    for (std::size_t i = 0; i < data.row_count; ++i) {
        data.columns[0].push_back(std::to_string(i));
        data.columns[1].push_back(
            i % 2 == 0 ? "red" : "blue"
        );

        if (task == lab5::Task::Classification) {
            data.columns[2].push_back(
                i < 6 ? "NO" : "YES"
            );
        } else {
            data.columns[2].push_back(
                std::to_string(i * 2)
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
    config.parameters.tree = {3, 2, 1};

    return config;
}

void test_final_evaluation(lab5::Task task) {
    const RawDataset train = make_dataset(task);
    const RawDataset test = make_dataset(task);

    const PreprocessingOptions preprocessing{};
    const auto model = make_model(task);

    const lab5::Scoring scoring =
        task == lab5::Task::Classification
            ? lab5::Scoring::F1
            : lab5::Scoring::MAE;

    const auto trained = lab5::train_final_model(
        train,
        "target",
        preprocessing,
        model,
        scoring
    );

    const auto first = lab5::predict_final_model(
        trained, test
    );

    const auto second = lab5::predict_final_model(
        trained, test
    );

    require(
        first.predicted == second.predicted,
        "Repeated prediction changed"
    );

    require(
        first.numeric_predictions ==
            second.numeric_predictions,
        "Numeric predictions changed"
    );

    require(
        first.actual.size() == test.row_count,
        "Wrong prediction count"
    );

    const auto result = lab5::evaluate_final_model(
        trained,
        train,
        test
    );

    const std::size_t expected_metrics =
        task == lab5::Task::Classification ? 4 : 3;

    require(
        result.metrics.size() == expected_metrics,
        "Incorrect metric count"
    );

    for (const auto& metric : result.metrics) {
        require(
            std::isfinite(metric.train) &&
            std::isfinite(metric.test),
            "Non-finite evaluation metric"
        );

        // Both sets are identical in this controlled test.
        require(
            std::abs(metric.train - metric.test) < 1e-10,
            "Equal datasets produced different metrics"
        );
    }

    const double score = lab5::score_final_model(
        trained, test
    );

    require(
        std::isfinite(score),
        "Non-finite scoring value"
    );

    const auto directory =
        std::filesystem::temp_directory_path();

    const auto metrics_path =
        directory / "lab5_evaluation_metrics_test.csv";

    const auto predictions_path =
        directory / "lab5_evaluation_predictions_test.csv";

    std::vector<std::uint64_t> indices(test.row_count);

    for (std::size_t i = 0; i < indices.size(); ++i) {
        indices[i] = static_cast<std::uint64_t>(i);
    }

    lab5::save_evaluation_metrics(
        result,
        metrics_path.string()
    );

    lab5::save_evaluation_predictions(
        result,
        indices,
        predictions_path.string()
    );

    require(
        std::filesystem::exists(metrics_path),
        "Metrics file was not created"
    );

    require(
        std::filesystem::exists(predictions_path),
        "Predictions file was not created"
    );

    std::filesystem::remove(metrics_path);
    std::filesystem::remove(predictions_path);
}

void test_unseen_class_rejected() {
    const RawDataset train = make_dataset(
        lab5::Task::Classification
    );

    const auto trained = lab5::train_final_model(
        train,
        "target",
        PreprocessingOptions{},
        make_model(lab5::Task::Classification),
        lab5::Scoring::F1
    );

    RawDataset test = train;
    test.columns[2][0] = "UNKNOWN";

    bool caught = false;

    try {
        (void)lab5::score_final_model(trained, test);
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    require(
        caught,
        "Unknown target class was accepted"
    );
}

} // namespace

int main() {
    try {
        test_final_evaluation(
            lab5::Task::Classification
        );

        test_final_evaluation(
            lab5::Task::Regression
        );

        test_unseen_class_rejected();

        std::cout
            << "All Lab 5 evaluation tests passed.\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "[FAIL] " << error.what() << '\n';

        return 1;
    }
}



