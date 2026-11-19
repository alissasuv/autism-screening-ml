#pragma once

#include "lab5_preprocessing.hpp"
#include "metrics.hpp"
#include "model_factory.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace lab5 {

struct TrainedEvaluation {
    Task task = Task::Classification;
    Scoring scoring = Scoring::F1;

    FittedRawPreprocessing preprocessing;

    std::unique_ptr<IClassifier> classifier;
    std::unique_ptr<IRegressor> regressor;

    // Fixed class names and numeric IDs learned from train.
    std::vector<std::string> class_names;
    std::map<std::string, double> class_ids;
};

struct PredictionResult {
    std::vector<std::string> actual;
    std::vector<std::string> predicted;
    std::vector<double> numeric_predictions;
};

struct MetricRow {
    std::string name;
    double train = 0.0;
    double test = 0.0;
};

struct FinalEvaluation {
    std::vector<MetricRow> metrics;
    PredictionResult test_predictions;
};

// Fit preprocessing and one new model on the complete train set.
TrainedEvaluation train_final_model(
    const RawDataset& train,
    const std::string& target_name,
    const PreprocessingOptions& preprocessing,
    const ModelConfig& model,
    Scoring scoring
);

// Predict using the previously trained preprocessing and model.
// The model and preprocessing are never fitted again here.
PredictionResult predict_final_model(
    const TrainedEvaluation& trained,
    const RawDataset& data
);

// Calculate the configured scoring metric on any evaluation set.
double score_final_model(
    const TrainedEvaluation& trained,
    const RawDataset& data
);

// Calculate every task metric separately for train and test.
FinalEvaluation evaluate_final_model(
    const TrainedEvaluation& trained,
    const RawDataset& train,
    const RawDataset& test
);

void save_evaluation_metrics(
    const FinalEvaluation& result,
    const std::string& filename
);

void save_evaluation_predictions(
    const FinalEvaluation& result,
    const std::vector<std::uint64_t>& source_indices,
    const std::string& filename
);

} // namespace lab5

