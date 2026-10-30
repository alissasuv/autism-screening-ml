#include "csv_reader.hpp"
#include "dataset.hpp"
#include "decision_stump.hpp"
#include "feature_type.hpp"
#include "model.hpp"
#include "preprocessing.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    try {
        const std::string input_path =
            argc > 1 ? argv[1] : "data/Autism-Adult-Data.csv";

        // Load and identify dataset columns.
        const RawDataset raw = read_csv(input_path);

        std::vector<FeatureType> types;
        types.reserve(raw.columns.size());

        for (const auto& column : raw.columns) {
            types.push_back(detect_feature_type(column));
        }

        const Dataset dataset = make_typed_dataset(raw, types);

        // Split the dataset before preprocessing.
        const DatasetSplit split = split_dataset(dataset, 0.8, 42);

        const PreprocessingOptions options;
        const PreprocessingState state =
            fit_preprocessing(split.train, options);

        const NumericMatrix train =
            transform_preprocessing(split.train, state);

        const NumericMatrix test =
            transform_preprocessing(split.test, state);

        const std::vector<double> train_target =
            encode_target(train.target_values);

        const std::vector<double> test_target =
            encode_target(test.target_values);

        if (train.cols != test.cols ||
            train.column_names != test.column_names) {
            throw std::runtime_error(
                "Train and test feature columns do not match"
            );
        }

        std::cout << "Decision Stump - Lab 2 Star\n\n";
        std::cout << "Total objects: " << dataset.row_count << '\n';
        std::cout << "Training objects: " << train.rows << '\n';
        std::cout << "Test objects: " << test.rows << '\n';
        std::cout << "Features: " << train.cols << '\n';

        // Demonstrate both required learning tasks.
        for (const SplitCriterion criterion :
             {SplitCriterion::Gini, SplitCriterion::Variance}) {

            std::unique_ptr<IModel> model =
                std::make_unique<DecisionStump>(criterion, 5);

            model->fit(train, train_target);
            const auto predictions = model->predict(test);

            // Access stump-specific information for demonstration.
            const auto& stump =
                static_cast<const DecisionStump&>(*model);

            std::cout << "\n-----------------------------\n";
            std::cout << (criterion == SplitCriterion::Gini
                ? "Classification (Gini)\n"
                : "Regression (Variance)\n");

            if (stump.has_split()) {
                std::cout << "Split feature: "
                          << train.column_names[
                                 stump.split_feature()
                             ] << '\n';

                std::cout << "Split threshold: "
                          << std::fixed
                          << std::setprecision(4)
                          << stump.split_threshold() << '\n';
            } else {
                std::cout << "No valid split found.\n";
            }

            std::cout << "Left prediction: "
                      << stump.left_prediction() << '\n';
            std::cout << "Right prediction: "
                      << stump.right_prediction() << '\n';

            std::cout << "\nFirst test predictions:\n";

            const std::size_t count =
                std::min<std::size_t>(10, predictions.size());

            for (std::size_t i = 0; i < count; ++i) {
                std::cout << "Object " << i + 1
                          << " | Predicted: " << predictions[i]
                          << " | Actual: " << test_target[i]
                          << '\n';
            }
        }

        return 0;

    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
