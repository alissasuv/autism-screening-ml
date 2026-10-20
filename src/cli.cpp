#include "cli.hpp"

#include <cxxopts.hpp>

#include <stdexcept>
#include <string>

namespace {

NumericImputation parse_imputation(const std::string& name) {
    if (name == "mean") {
        return NumericImputation::Mean;
    }
    if (name == "median") {
        return NumericImputation::Median;
    }
    if (name == "zero") {
        return NumericImputation::Zero;
    }

    throw std::invalid_argument(
        "Invalid imputation strategy: " + name +
        ". Choose mean, median or zero."
    );
}

NumericScaling parse_scaling(const std::string& name) {
    if (name == "min-max") {
        return NumericScaling::MinMax;
    }
    if (name == "standard") {
        return NumericScaling::Standard;
    }
    if (name == "robust") {
        return NumericScaling::Robust;
    }

    throw std::invalid_argument(
        "Invalid scaling strategy: " + name +
        ". Choose min-max, standard or robust."
    );
}

CategoryEncoding parse_encoding(const std::string& name) {
    if (name == "one-hot") {
        return CategoryEncoding::OneHot;
    }
    if (name == "ordinal") {
        return CategoryEncoding::Ordinal;
    }

    throw std::invalid_argument(
        "Invalid encoding strategy: " + name +
        ". Choose one-hot or ordinal."
    );
}

} // namespace

CliOptions parse_cli(int argc, char** argv) {
    cxxopts::Options options(
        "autism_screening",
        "Tabular data analysis and preprocessing in C++20"
    );

    options.add_options()
        ("i,input", "Input CSV file",
            cxxopts::value<std::string>()
                ->default_value("data/Autism-Adult-Data.csv"))

        ("o,output", "Output binary file",
            cxxopts::value<std::string>()
                ->default_value("data/processed.bin"))

        ("b,bins", "Number of histogram bins",
            cxxopts::value<std::size_t>()
                ->default_value("10"))

        ("imputation", "Numeric imputation: mean, median, zero",
            cxxopts::value<std::string>()
                ->default_value("median"))

        ("scaling", "Numeric scaling: min-max, standard, robust",
            cxxopts::value<std::string>()
                ->default_value("min-max"))

        ("encoding", "Category encoding: one-hot, ordinal",
            cxxopts::value<std::string>()
                ->default_value("one-hot"))

        ("h,help", "Show help");

    const auto result = options.parse(argc, argv);

    CliOptions config;

    if (result.count("help") != 0) {
        config.show_help = true;
        config.help_text = options.help();
        return config;
    }

    config.input_path = result["input"].as<std::string>();
    config.output_path = result["output"].as<std::string>();
    config.histogram_bins = result["bins"].as<std::size_t>();

    if (config.input_path.empty()) {
        throw std::invalid_argument(
            "Input path cannot be empty"
        );
    }

    if (config.output_path.empty()) {
        throw std::invalid_argument(
            "Output path cannot be empty"
        );
    }

    if (config.histogram_bins == 0 ||
        config.histogram_bins > 1000) {
        throw std::invalid_argument(
            "Histogram bins must be between 1 and 1000"
        );
    }

    config.preprocessing.imputation =
        parse_imputation(result["imputation"].as<std::string>());

    config.preprocessing.scaling =
        parse_scaling(result["scaling"].as<std::string>());

    config.preprocessing.encoding =
        parse_encoding(result["encoding"].as<std::string>());

    return config;
}