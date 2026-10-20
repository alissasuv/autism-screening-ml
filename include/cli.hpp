#pragma once

#include "preprocessing.hpp"

#include <cstddef>
#include <string>

struct CliOptions {
    std::string input_path = "data/Autism-Adult-Data.csv";
    std::string output_path = "data/processed.bin";

    std::size_t histogram_bins = 10;

    PreprocessingOptions preprocessing;

    bool show_help = false;
    std::string help_text;
};

CliOptions parse_cli(int argc, char** argv);