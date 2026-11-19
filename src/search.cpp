#include "search.hpp"

#include "validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lab5 {
namespace {

bool known_parameter(const std::string& name) {
    return name == "n_estimators" ||
           name == "max_depth" ||
           name == "min_samples_split" ||
           name == "min_samples_leaf" ||
           name == "max_features" ||
           name == "max_samples" ||
           name == "bootstrap" ||
           name == "random_state" ||
           name == "n_jobs";
}

void check_parameters(const YAML::Node& params) {
    if (!params || !params.IsMap()) {
        throw std::invalid_argument(
            "model.params must be a YAML mapping"
        );
    }

    for (const auto& item : params) {
        const std::string name =
            item.first.as<std::string>();

        if (!known_parameter(name)) {
            throw std::invalid_argument(
                "Unknown forest parameter: " + name
            );
        }
    }
}

std::size_t positive_size(
    const YAML::Node& value,
    const std::string& name,
    std::size_t minimum = 1
) {
    if (!value || !value.IsScalar()) {
        throw std::invalid_argument(
            "Missing integer parameter: " + name
        );
    }

    const auto number = value.as<std::int64_t>();

    if (number < 0 ||
        static_cast<std::uint64_t>(number) < minimum) {
        throw std::invalid_argument(
            "Invalid integer parameter: " + name
        );
    }

    return static_cast<std::size_t>(number);
}

std::uint32_t read_seed(
    const YAML::Node& value,
    const std::string& name
) {
    if (!value || !value.IsScalar()) {
        throw std::invalid_argument(
            "Missing seed: " + name
        );
    }

    const auto number = value.as<std::int64_t>();

    if (number < 0 ||
        number > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "Seed out of range: " + name
        );
    }

    return static_cast<std::uint32_t>(number);
}

YAML::Node require_map(
    const YAML::Node& parent,
    const std::string& name
) {
    const YAML::Node value = parent[name];

    if (!value || !value.IsMap()) {
        throw std::invalid_argument(
            "Expected YAML mapping: " + name
        );
    }

    return value;
}

std::string require_string(
    const YAML::Node& parent,
    const std::string& name
) {
    const YAML::Node value = parent[name];

    if (!value || !value.IsScalar()) {
        throw std::invalid_argument(
            "Expected YAML scalar: " + name
        );
    }

    return value.as<std::string>();
}

std::vector<YAML::Node> expand_values(
    const YAML::Node& specification
) {
    if (!specification.IsMap()) {
        throw std::invalid_argument(
            "Search parameter specification must be a map"
        );
    }

    std::vector<YAML::Node> result;

    if (specification["values"]) {
        if (specification.size() != 1 ||
            !specification["values"].IsSequence() ||
            specification["values"].size() == 0) {
            throw std::invalid_argument(
                "values must be a nonempty sequence"
            );
        }

        for (const auto& value : specification["values"]) {
            if (!value.IsScalar() && !value.IsNull()) {
                throw std::invalid_argument(
                    "Search values must be scalar or null"
                );
            }

            result.push_back(YAML::Clone(value));
        }

        return result;
    }

    const YAML::Node start_node = specification["start"];
    const YAML::Node stop_node = specification["stop"];

    if (!start_node || !stop_node) {
        throw std::invalid_argument(
            "Range requires start and stop"
        );
    }

    if (specification["step"]) {
        if (specification.size() != 3) {
            throw std::invalid_argument(
                "Integer range must use start, stop, step"
            );
        }

        const auto start = start_node.as<std::int64_t>();
        const auto stop = stop_node.as<std::int64_t>();
        const auto step =
            specification["step"].as<std::int64_t>();

        if (step <= 0 || start > stop) {
            throw std::invalid_argument(
                "Invalid integer search range"
            );
        }

        for (auto value = start; value <= stop;) {
            result.push_back(YAML::Node(value));

            // Avoid signed overflow.
            if (stop - value < step) {
                break;
            }

            value += step;
        }

        return result;
    }

    if (!specification["count"] ||
        !specification["scale"] ||
        specification.size() != 4) {
        throw std::invalid_argument(
            "Float range requires start, stop, count, scale"
        );
    }

    const double start = start_node.as<double>();
    const double stop = stop_node.as<double>();

    const std::size_t count =
        positive_size(specification["count"], "count", 2);

    const std::string scale =
        specification["scale"].as<std::string>();

    if (!std::isfinite(start) ||
        !std::isfinite(stop) ||
        start > stop ||
        (scale != "linear" && scale != "log") ||
        (scale == "log" && start <= 0.0)) {
        throw std::invalid_argument(
            "Invalid floating-point search range"
        );
    }

    for (std::size_t i = 0; i < count; ++i) {
        const double fraction =
            static_cast<double>(i) /
            static_cast<double>(count - 1);

        double value = 0.0;

        if (scale == "linear") {
            value = start + fraction * (stop - start);
        } else {
            value = std::exp(
                std::log(start) +
                fraction * (std::log(stop) - std::log(start))
            );
        }

        if (i == 0) {
            value = start;
        } else if (i + 1 == count) {
            value = stop;
        }

        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Non-finite search grid value"
            );
        }

        result.push_back(YAML::Node(value));
    }

    return result;
}

std::string csv_quote(const std::string& value) {
    std::string result = "\"";

    for (const char ch : value) {
        if (ch == '"') {
            result += "\"\"";
        } else {
            result += ch;
        }
    }

    result += '"';
    return result;
}

std::size_t candidate_count(const SearchConfig& config) {
    std::size_t count = 1;

    for (const auto& axis : config.axes) {
        if (axis.values.empty() ||
            count > 100000 / axis.values.size()) {
            throw std::invalid_argument(
                "Search grid is empty or excessively large"
            );
        }

        count *= axis.values.size();
    }

    return count;
}

void build_candidates(
    const SearchConfig& config,
    std::size_t axis_index,
    const YAML::Node& current,
    std::vector<YAML::Node>& output
) {
    if (axis_index == config.axes.size()) {
        output.push_back(YAML::Clone(current));
        return;
    }

    const auto& axis = config.axes[axis_index];

    for (const auto& value : axis.values) {
        YAML::Node next = YAML::Clone(current);
        next[axis.name] = YAML::Clone(value);

        build_candidates(
            config,
            axis_index + 1,
            next,
            output
        );
    }
}

} // namespace

std::string task_name(Task task) {
    switch (task) {
        case Task::Classification:
            return "classification";

        case Task::Regression:
            return "regression";
    }

    throw std::invalid_argument("Unknown task");
}

ModelConfig parse_model_config(
    Task task,
    const YAML::Node& params
) {
    check_parameters(params);

    ModelConfig result;
    result.task = task;

    auto& forest = result.parameters;

    forest = task == Task::Classification
        ? classifier_default_parameters()
        : RandomForestParameters{};

    if (params["n_estimators"]) {
        forest.n_estimators = positive_size(
            params["n_estimators"], "n_estimators"
        );
    }

    if (params["bootstrap"]) {
        forest.bootstrap =
            params["bootstrap"].as<bool>();
    }

    if (params["max_samples"] &&
        !params["max_samples"].IsNull()) {
        forest.max_samples = positive_size(
            params["max_samples"], "max_samples"
        );
    }

    if (!forest.bootstrap && forest.max_samples.has_value()) {
        throw std::invalid_argument(
            "max_samples cannot be set when bootstrap=false"
        );
    }

    if (params["max_features"]) {
        const std::string value =
            params["max_features"].as<std::string>();

        if (value == "sqrt") {
            forest.max_features = {FeatureMode::Sqrt, 1};
        } else if (value == "all") {
            forest.max_features = {FeatureMode::All, 1};
        } else {
            forest.max_features = {
                FeatureMode::Count,
                positive_size(
                    params["max_features"],
                    "max_features"
                )
            };
        }
    }

    if (params["max_depth"]) {
        if (params["max_depth"].IsNull()) {
            forest.tree.max_depth = std::nullopt;
        } else {
            forest.tree.max_depth = positive_size(
                params["max_depth"], "max_depth"
            );
        }
    }

    if (params["min_samples_split"]) {
        forest.tree.min_samples_split = positive_size(
            params["min_samples_split"],
            "min_samples_split",
            2
        );
    }

    if (params["min_samples_leaf"]) {
        forest.tree.min_samples_leaf = positive_size(
            params["min_samples_leaf"],
            "min_samples_leaf"
        );
    }

    if (params["random_state"]) {
        forest.random_state = read_seed(
            params["random_state"], "random_state"
        );
    }

    if (params["n_jobs"]) {
        forest.n_jobs = positive_size(
            params["n_jobs"], "n_jobs"
        );
    }

    return result;
}

SearchConfig load_search_config(
    const std::string& filename
) {
    const YAML::Node root = YAML::LoadFile(filename);

    if (!root.IsMap()) {
        throw std::invalid_argument(
            "Search configuration must be a YAML mapping"
        );
    }

    SearchConfig config;

    const std::string task = require_string(root, "task");

    if (task == "classification") {
        config.task = Task::Classification;
    } else if (task == "regression") {
        config.task = Task::Regression;
    } else {
        throw std::invalid_argument(
            "Unknown task: " + task
        );
    }

    config.scoring = parse_scoring(
        require_string(root, "scoring")
    );

    validate_scoring(
        config.scoring,
        config.task == Task::Classification
    );

    const YAML::Node model = require_map(root, "model");

    if (require_string(model, "type") != "random_forest") {
        throw std::invalid_argument(
            "Only random_forest is supported in Lab 5 star"
        );
    }

    config.base_params = YAML::Clone(
        require_map(model, "params")
    );

    (void)parse_model_config(
        config.task, config.base_params
    );

    const YAML::Node cv = require_map(root, "cv");

    config.n_splits = positive_size(
        cv["n_splits"], "cv.n_splits", 2
    );

    config.cv_seed = read_seed(
        cv["random_state"], "cv.random_state"
    );

    const YAML::Node search = require_map(root, "search");

    if (require_string(search, "method") != "grid") {
        throw std::invalid_argument(
            "This implementation supports grid search only"
        );
    }

    const YAML::Node parameters =
        require_map(search, "parameters");

    for (const auto& item : parameters) {
        SearchAxis axis;
        axis.name = item.first.as<std::string>();

        if (!known_parameter(axis.name)) {
            throw std::invalid_argument(
                "Unknown search parameter: " + axis.name
            );
        }

        axis.values = expand_values(item.second);

        for (const auto& value : axis.values) {
            YAML::Node candidate = YAML::Clone(
                config.base_params
            );

            candidate[axis.name] = YAML::Clone(value);

            // Validate each axis value before training.
            (void)parse_model_config(
                config.task, candidate
            );
        }

        config.axes.push_back(std::move(axis));
    }

    (void)candidate_count(config);
    return config;
}

SearchResult run_grid_search(
    const RawDataset& train,
    const std::string& target_name,
    const PreprocessingOptions& preprocessing,
    const SearchConfig& config
) {
    const auto folds = make_folds(
        train,
        target_name,
        config.task,
        config.n_splits,
        config.cv_seed
    );

    std::vector<YAML::Node> candidates;

    build_candidates(
        config,
        0,
        config.base_params,
        candidates
    );

    if (candidates.empty()) {
        throw std::runtime_error(
            "No grid-search candidates"
        );
    }

    const std::size_t total_fits =
        candidates.size() * config.n_splits;

    std::cout
        << "Candidates: " << candidates.size() << '\n'
        << "CV folds: " << config.n_splits << '\n'
        << "Planned fits: " << total_fits << '\n';

    SearchResult result;
    result.records.reserve(candidates.size());

    for (std::size_t i = 0; i < candidates.size(); ++i) {
        SearchRecord record;
        record.params = YAML::Clone(candidates[i]);

        std::cout
            << "Candidate " << (i + 1)
            << "/" << candidates.size()
            << " ... " << std::flush;

        try {
            const ModelConfig model = parse_model_config(
                config.task, record.params
            );

            const CrossValidationResult cv =
                cross_validate(
                    train,
                    target_name,
                    preprocessing,
                    model,
                    config.scoring,
                    folds
                );

            record.fold_scores = cv.fold_scores;
            record.mean = cv.mean;
            record.stddev = cv.stddev;
            record.valid = true;

            std::cout
                << "mean=" << std::setprecision(6)
                << record.mean << '\n';

        } catch (const std::invalid_argument& error) {
            // Candidate-specific incompatibility with training
            // folds must not invalidate other candidates.
            record.error = error.what();

            std::cout
                << "invalid: " << record.error << '\n';
        }

        if (record.valid &&
            (!result.has_best ||
             is_better_score(
                 record.mean,
                 result.records[result.best_index].mean,
                 config.scoring
             ))) {
            result.best_index = result.records.size();
            result.has_best = true;
        }

        result.records.push_back(std::move(record));
    }

    if (!result.has_best) {
        throw std::runtime_error(
            "Grid search found no valid candidates"
        );
    }

    return result;
}

void save_search_results(
    const SearchResult& result,
    const SearchConfig& config,
    const std::string& filename
) {
    std::ofstream out(filename);

    if (!out) {
        throw std::runtime_error(
            "Cannot create search results: " + filename
        );
    }

    out << "candidate,valid,parameters";

    for (std::size_t fold = 0; fold < config.n_splits; ++fold) {
        out << ",fold_" << (fold + 1);
    }

    out << ",mean,stddev,error\n";
    out << std::setprecision(17);

    for (std::size_t i = 0; i < result.records.size(); ++i) {
        const auto& record = result.records[i];

        out << (i + 1) << ','
            << (record.valid ? "true" : "false") << ','
            << csv_quote(YAML::Dump(record.params));

        for (std::size_t fold = 0;
             fold < config.n_splits;
             ++fold) {
            out << ',';

            if (fold < record.fold_scores.size()) {
                out << record.fold_scores[fold];
            }
        }

        out << ',';

        if (record.valid) {
            out << record.mean;
        }

        out << ',';

        if (record.valid) {
            out << record.stddev;
        }

        out << ','
            << csv_quote(record.error)
            << '\n';
    }

    if (!out) {
        throw std::runtime_error(
            "Failed to write search results"
        );
    }
}

void save_best_model_config(
    const SearchResult& result,
    const SearchConfig& config,
    const std::string& filename
) {
    if (!result.has_best ||
        result.best_index >= result.records.size()) {
        throw std::invalid_argument(
            "No best model is available"
        );
    }

    YAML::Node output(YAML::NodeType::Map);

    output["task"] = task_name(config.task);
    output["scoring"] = scoring_name(config.scoring);

    output["model"]["type"] = "random_forest";
    output["model"]["params"] = YAML::Clone(
        result.records[result.best_index].params
    );

    // Record defaults explicitly for reproducibility.
    const auto parameters = parse_model_config(
        config.task,
        output["model"]["params"]
    ).parameters;

    YAML::Node saved = output["model"]["params"];

    saved["n_estimators"] = parameters.n_estimators;
    saved["bootstrap"] = parameters.bootstrap;

    if (parameters.max_samples.has_value()) {
        saved["max_samples"] = *parameters.max_samples;
    } else {
        saved["max_samples"] = YAML::Node(YAML::NodeType::Null);
    }

    switch (parameters.max_features.mode) {
        case FeatureMode::All:
            saved["max_features"] = "all";
            break;

        case FeatureMode::Sqrt:
            saved["max_features"] = "sqrt";
            break;

        case FeatureMode::Count:
            saved["max_features"] =
                parameters.max_features.count;
            break;
    }

    if (parameters.tree.max_depth.has_value()) {
        saved["max_depth"] = *parameters.tree.max_depth;
    } else {
        saved["max_depth"] = YAML::Node(YAML::NodeType::Null);
    }

    saved["min_samples_split"] =
        parameters.tree.min_samples_split;

    saved["min_samples_leaf"] =
        parameters.tree.min_samples_leaf;

    saved["random_state"] = parameters.random_state;
    saved["n_jobs"] = parameters.n_jobs;

    std::ofstream out(filename);

    if (!out) {
        throw std::runtime_error(
            "Cannot create best model configuration: " + filename
        );
    }

    out << output << '\n';

    if (!out) {
        throw std::runtime_error(
            "Failed to save best model configuration"
        );
    }
}

} // namespace lab5


