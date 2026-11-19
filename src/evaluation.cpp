#include "evaluation.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#include <sstream>

namespace lab5 {
namespace {

double parse_number(const std::string& text) {
    double value = 0.0;

    const auto [end, error] = std::from_chars(
        text.data(),
        text.data() + text.size(),
        value
    );

    if (error != std::errc{} ||
        end != text.data() + text.size() ||
        !std::isfinite(value)) {
        throw std::invalid_argument(
            "Invalid numeric target: " + text
        );
    }

    return value;
}

std::vector<double> numeric_targets(
    const std::vector<std::string>& values
) {
    std::vector<double> result;
    result.reserve(values.size());

    for (const auto& text : values) {
        result.push_back(parse_number(text));
    }

    return result;
}

std::vector<double> encode_classes(
    const std::vector<std::string>& values,
    const std::map<std::string, double>& mapping
) {
    std::vector<double> result;
    result.reserve(values.size());

    for (const auto& value : values) {
        const auto it = mapping.find(value);

        if (it == mapping.end()) {
            throw std::invalid_argument(
                "Unknown classification target: " + value
            );
        }

        result.push_back(it->second);
    }

    return result;
}

std::vector<double> class_ids(
    const TrainedEvaluation& trained
) {
    std::vector<double> ids;
    ids.reserve(trained.class_names.size());

    for (std::size_t i = 0;
         i < trained.class_names.size();
         ++i) {
        ids.push_back(static_cast<double>(i));
    }

    return ids;
}

std::string class_name(
    double value,
    const TrainedEvaluation& trained
) {
    if (!std::isfinite(value) ||
        value < 0.0 ||
        std::floor(value) != value ||
        value >= static_cast<double>(
            trained.class_names.size()
        )) {
        throw std::runtime_error(
            "Model predicted an unknown class ID"
        );
    }

    return trained.class_names[
        static_cast<std::size_t>(value)
    ];
}

std::string number_string(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "Cannot serialize a non-finite prediction"
        );
    }

    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

std::string csv_escape(const std::string& text) {
    std::string result = "\"";

    for (const char ch : text) {
        if (ch == '"') {
            result += "\"\"";
        } else {
            result += ch;
        }
    }

    result += '"';
    return result;
}

std::vector<std::pair<std::string, double>> all_metrics(
    const TrainedEvaluation& trained,
    const PredictionResult& prediction
) {
    if (trained.task == Task::Classification) {
        const auto actual = encode_classes(
            prediction.actual,
            trained.class_ids
        );

        const auto classes = class_ids(trained);

        const ClassificationMetrics metrics =
            classification_metrics(
                actual,
                prediction.numeric_predictions,
                classes
            );

        return {
            {"accuracy", metrics.accuracy},
            {"precision", metrics.precision},
            {"recall", metrics.recall},
            {"f1", metrics.f1}
        };
    }

    const auto actual = numeric_targets(
        prediction.actual
    );

    const RegressionMetrics metrics =
        regression_metrics(
            actual,
            prediction.numeric_predictions
        );

    return {
        {"mse", metrics.mse},
        {"mae", metrics.mae},
        {"r2", metrics.r2}
    };
}

} // namespace

TrainedEvaluation train_final_model(
    const RawDataset& train,
    const std::string& target_name,
    const PreprocessingOptions& preprocessing,
    const ModelConfig& model,
    Scoring scoring
) {
    validate_scoring(
        scoring,
        model.task == Task::Classification
    );

    TrainedEvaluation trained;
    trained.task = model.task;
    trained.scoring = scoring;

    trained.preprocessing = fit_raw_preprocessing(
        train,
        target_name,
        preprocessing
    );

    const NumericMatrix features =
        transform_raw_preprocessing(
            train,
            trained.preprocessing
        );

    if (features.rows == 0 || features.cols == 0) {
        throw std::invalid_argument(
            "Final training requires nonempty features"
        );
    }

    if (trained.task == Task::Classification) {
        // std::map sorts class names deterministically.
        for (const auto& label : features.target_values) {
            trained.class_ids.emplace(label, 0.0);
        }

        for (auto& [name, id] : trained.class_ids) {
            id = static_cast<double>(
                trained.class_names.size()
            );
            trained.class_names.push_back(name);
        }

        const auto targets = encode_classes(
            features.target_values,
            trained.class_ids
        );

        trained.classifier = create_classifier(model);
        trained.classifier->fit(features, targets);

    } else {
        const auto targets = numeric_targets(
            features.target_values
        );

        trained.regressor = create_regressor(model);
        trained.regressor->fit(features, targets);
    }

    return trained;
}

PredictionResult predict_final_model(
    const TrainedEvaluation& trained,
    const RawDataset& data
) {
    const NumericMatrix features =
        transform_raw_preprocessing(
            data,
            trained.preprocessing
        );

    PredictionResult result;
    result.actual = features.target_values;

    if (trained.task == Task::Classification) {
        if (!trained.classifier) {
            throw std::logic_error(
                "Classifier has not been trained"
            );
        }

        result.numeric_predictions =
            trained.classifier->predict(features);

        result.predicted.reserve(
            result.numeric_predictions.size()
        );

        for (const double prediction :
             result.numeric_predictions) {
            result.predicted.push_back(
                class_name(prediction, trained)
            );
        }

    } else {
        if (!trained.regressor) {
            throw std::logic_error(
                "Regressor has not been trained"
            );
        }

        result.numeric_predictions =
            trained.regressor->predict(features);

        result.predicted.reserve(
            result.numeric_predictions.size()
        );

        for (const double prediction :
             result.numeric_predictions) {
            result.predicted.push_back(
                number_string(prediction)
            );
        }
    }

    if (result.actual.size() != result.predicted.size()) {
        throw std::runtime_error(
            "Prediction count does not match dataset rows"
        );
    }

    return result;
}

double score_final_model(
    const TrainedEvaluation& trained,
    const RawDataset& data
) {
    const PredictionResult prediction =
        predict_final_model(trained, data);

    if (trained.task == Task::Classification) {
        const auto actual = encode_classes(
            prediction.actual,
            trained.class_ids
        );

        const auto classes = class_ids(trained);

        return classification_scoring(
            trained.scoring,
            actual,
            prediction.numeric_predictions,
            classes
        );
    }

    const auto actual = numeric_targets(
        prediction.actual
    );

    return regression_scoring(
        trained.scoring,
        actual,
        prediction.numeric_predictions
    );
}

FinalEvaluation evaluate_final_model(
    const TrainedEvaluation& trained,
    const RawDataset& train,
    const RawDataset& test
) {
    const PredictionResult train_predictions =
        predict_final_model(trained, train);

    const PredictionResult test_predictions =
        predict_final_model(trained, test);

    const auto train_metrics = all_metrics(
        trained,
        train_predictions
    );

    const auto test_metrics = all_metrics(
        trained,
        test_predictions
    );

    if (train_metrics.size() != test_metrics.size()) {
        throw std::logic_error(
            "Training and test metric counts differ"
        );
    }

    FinalEvaluation result;
    result.test_predictions = test_predictions;

    for (std::size_t i = 0;
         i < train_metrics.size();
         ++i) {
        if (train_metrics[i].first !=
            test_metrics[i].first) {
            throw std::logic_error(
                "Metric order mismatch"
            );
        }

        result.metrics.push_back({
            train_metrics[i].first,
            train_metrics[i].second,
            test_metrics[i].second
        });
    }

    return result;
}

void save_evaluation_metrics(
    const FinalEvaluation& result,
    const std::string& filename
) {
    std::ofstream out(filename);

    if (!out) {
        throw std::runtime_error(
            "Cannot create metrics CSV: " + filename
        );
    }

    out << "metric,train,test\n";
    out << std::setprecision(17);

    for (const auto& metric : result.metrics) {
        out << csv_escape(metric.name) << ','
            << metric.train << ','
            << metric.test << '\n';
    }

    if (!out) {
        throw std::runtime_error(
            "Failed to write metrics CSV"
        );
    }
}

void save_evaluation_predictions(
    const FinalEvaluation& result,
    const std::vector<std::uint64_t>& source_indices,
    const std::string& filename
) {
    const auto& predictions = result.test_predictions;

    if (source_indices.size() != predictions.actual.size() ||
        predictions.actual.size() != predictions.predicted.size()) {
        throw std::invalid_argument(
            "Prediction and source-index counts differ"
        );
    }

    std::ofstream out(filename);

    if (!out) {
        throw std::runtime_error(
            "Cannot create predictions CSV: " + filename
        );
    }

    out << "test_row,source_row,actual,predicted\n";

    for (std::size_t i = 0;
         i < predictions.actual.size();
         ++i) {
        out << i << ','
            << source_indices[i] << ','
            << csv_escape(predictions.actual[i]) << ','
            << csv_escape(predictions.predicted[i]) << '\n';
    }

    if (!out) {
        throw std::runtime_error(
            "Failed to write predictions CSV"
        );
    }
}

} // namespace lab5


