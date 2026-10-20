#pragma once

#include "preprocessing.hpp"

#include <string>

void save_binary_matrix(
    const NumericMatrix& matrix,
    const std::string& filename
);

NumericMatrix load_binary_matrix(
    const std::string& filename
);