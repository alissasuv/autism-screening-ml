#include "csv_reader.hpp"
#include "lab5_data.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstdlib>

namespace {

struct Arguments {
    std::string input;
    std::string target = "Class/ASD";
    std::string task = "classification";
    std::string train_output = "data/train.bin";
    std::string test_output = "data/test.bin";
    std::string stats_output = "results/dataset_stats.json";

    double test_size = 0.2;
    std::uint32_t random_state = 42;
};

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  prepare_data --input <csv>"
        << " [--target <name>]"
        << " [--task classification]"
        << " [--train-output <file>]"
        << " [--test-output <file>]"
        << " [--stats-output <file>]"
        << " [--test-size <fraction>]"
        << " [--random-state <seed>]\n";
}

double parse_double(const std::string& value) {
    std::size_t position = 0;
    const double result = std::stod(value, &position);

    if (position != value.size() || !std::isfinite(result)) {
        throw std::invalid_argument(
            "Invalid floating-point argument: " + value
        );
    }

    return result;
}

std::uint32_t parse_seed(const std::string& value) {
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument(
            "Invalid random_state: " + value
        );
    }

    std::size_t position = 0;
    const unsigned long long number = std::stoull(
        value, &position
    );

    if (position != value.size() ||
        number > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "Invalid random_state: " + value
        );
    }

    return static_cast<std::uint32_t>(number);
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];

        if (key == "--help" || key == "-h") {
            print_usage();
            std::exit(0);
        }

        if (i + 1 >= argc) {
            throw std::invalid_argument(
                "Missing value for argument: " + key
            );
        }

        const std::string value = argv[++i];

        if (key == "--input") {
            args.input = value;
        } else if (key == "--target") {
            args.target = value;
        } else if (key == "--task") {
            args.task = value;
        } else if (key == "--train-output") {
            args.train_output = value;
        } else if (key == "--test-output") {
            args.test_output = value;
        } else if (key == "--stats-output") {
            args.stats_output = value;
        } else if (key == "--test-size") {
            args.test_size = parse_double(value);
        } else if (key == "--random-state") {
            args.random_state = parse_seed(value);
        } else {
            throw std::invalid_argument(
                "Unknown argument: " + key
            );
        }
    }

    if (args.input.empty()) {
        throw std::invalid_argument(
            "Required argument: --input"
        );
    }

    if (args.task != "classification") {
        throw std::invalid_argument(
            "This prepare_data version supports classification only"
        );
    }

    if (!(args.test_size > 0.0 && args.test_size < 1.0)) {
        throw std::invalid_argument(
            "test-size must be between 0 and 1"
        );
    }

    return args;
}

void ensure_parent_directory(const std::string& filename) {
    const auto parent = std::filesystem::path(filename).parent_path();

    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);

        std::cout << "Lab 5 - Prepare Data\n";
        std::cout << "Input: " << args.input << '\n';
        std::cout << "Target: " << args.target << '\n';

        const RawDataset raw = read_csv(args.input);

        const PreparedSplit split = make_stratified_split(
            raw,
            args.target,
            args.test_size,
            args.random_state
        );

        ensure_parent_directory(args.train_output);
        ensure_parent_directory(args.test_output);
        ensure_parent_directory(args.stats_output);

        save_prepared_dataset(
            split.train,
            args.train_output
        );

        save_prepared_dataset(
            split.test,
            args.test_output
        );

        save_dataset_statistics(
            raw,
            split,
            args.target,
            args.test_size,
            args.random_state,
            args.stats_output
        );

        // Verify that the saved binary files can be loaded.
        const auto restored_train =
            load_prepared_dataset(args.train_output);

        const auto restored_test =
            load_prepared_dataset(args.test_output);

        std::cout << "Total rows: " << raw.row_count << '\n';
        std::cout << "Train rows: "
                  << restored_train.data.row_count << '\n';
        std::cout << "Test rows: "
                  << restored_test.data.row_count << '\n';

        std::cout << "Train saved: "
                  << args.train_output << '\n';
        std::cout << "Test saved: "
                  << args.test_output << '\n';
        std::cout << "Statistics saved: "
                  << args.stats_output << '\n';

        std::cout << "Preparation completed successfully.\n";
        return 0;

    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}


