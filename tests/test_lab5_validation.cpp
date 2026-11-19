#include "model_factory.hpp"
#include "validation.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <set>
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

RawDataset example(lab5::Task task) {
    RawDataset data;

    data.row_count = 12;
    data.column_names = {
        "x",
        "category",
        "response"
    };

    data.columns.resize(3);

    for (std::size_t i = 0; i < 12; ++i) {
        data.columns[0].push_back(
            std::to_string(i)
        );

        data.columns[1].push_back(
            i % 2 ? "blue" : "red"
        );

        data.columns[2].push_back(
            task == lab5::Task::Classification
                ? (i < 6 ? "NO" : "YES")
                : std::to_string(
                    static_cast<double>(i * i) / 2.0
                )
        );
    }

    return data;
}

lab5::ModelConfig model(lab5::Task task) {
    lab5::ModelConfig config;

    config.task = task;
    config.parameters.n_estimators = 3;
    config.parameters.n_jobs = 1;
    config.parameters.random_state = 42;
    config.parameters.bootstrap = false;

    config.parameters.max_features = {
        FeatureMode::All, 1
    };

    config.parameters.tree = {
        3, 2, 1
    };

    return config;
}

void test_folds(lab5::Task task) {
    const RawDataset data = example(task);

    const auto first = lab5::make_folds(
        data,
        "response",
        task,
        3,
        17
    );

    const auto second = lab5::make_folds(
        data,
        "response",
        task,
        3,
        17
    );

    require(
        first.size() == 3,
        "Wrong number of folds"
    );

    std::set<std::size_t> seen;

    for (std::size_t f = 0; f < first.size(); ++f) {
        require(
            first[f].validation_indices ==
                second[f].validation_indices,
            "Non-reproducible fold indices"
        );

        require(
            first[f].train_indices ==
                second[f].train_indices,
            "Non-reproducible training indices"
        );

        for (const auto index :
             first[f].validation_indices) {
            require(
                seen.insert(index).second,
                "Repeated validation row"
            );
        }

        if (task == lab5::Task::Classification) {
            std::size_t no = 0;
            std::size_t yes = 0;

            for (const auto index :
                 first[f].validation_indices) {
                if (*data.columns[2][index] == "NO") {
                    ++no;
                } else {
                    ++yes;
                }
            }

            require(
                no == 2 && yes == 2,
                "Stratified fold has wrong class counts"
            );
        }
    }

    require(
        seen.size() == data.row_count,
        "Fold coverage is incomplete"
    );
}

void test_cross_validation(lab5::Task task) {
    const RawDataset data = example(task);

    const auto folds = lab5::make_folds(
        data,
        "response",
        task,
        3,
        17
    );

    const PreprocessingOptions preprocessing{};
    const auto config = model(task);

    const auto scoring =
        task == lab5::Task::Classification
            ? lab5::Scoring::F1
            : lab5::Scoring::MAE;

    const auto first = lab5::cross_validate(
        data,
        "response",
        preprocessing,
        config,
        scoring,
        folds
    );

    const auto second = lab5::cross_validate(
        data,
        "response",
        preprocessing,
        config,
        scoring,
        folds
    );

    require(
        first.fold_scores.size() == 3,
        "Expected a score per fold"
    );

    require(
        first.fold_scores == second.fold_scores,
        "CV scores are not reproducible"
    );

    require(
        first.stddev >= 0 && std::isfinite(first.mean),
        "Invalid CV summary"
    );

    for (const double score : first.fold_scores) {
        require(
            std::isfinite(score),
            "Non-finite fold score"
        );

        if (task == lab5::Task::Classification) {
            require(
                score >= 0 && score <= 1,
                "Classification F1 outside [0,1]"
            );
        } else {
            require(
                score >= 0,
                "Negative MAE"
            );
        }
    }
}

void test_factory_errors() {
    const auto config = model(
        lab5::Task::Classification
    );

    bool caught = false;

    try {
        (void)lab5::create_regressor(config);
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    require(
        caught,
        "Factory allowed the wrong task"
    );
}

void test_small_class_rejected() {
    auto data = example(
        lab5::Task::Classification
    );

    data.columns[2][0] = "RARE";

    bool caught = false;

    try {
        (void)lab5::make_folds(
            data,
            "response",
            lab5::Task::Classification,
            3,
            17
        );
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    require(
        caught,
        "Accepted a class smaller than n_splits"
    );
}

} // namespace

int main() {
    try {
        test_folds(lab5::Task::Classification);
        test_folds(lab5::Task::Regression);

        test_cross_validation(
            lab5::Task::Classification
        );

        test_cross_validation(
            lab5::Task::Regression
        );

        test_factory_errors();
        test_small_class_rejected();

        std::cout
            << "All Lab 5 validation tests passed.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "[FAIL] " << error.what() << '\n';

        return 1;
    }
}

