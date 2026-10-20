#include "binary_io.hpp"

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char magic[4] = {'A', 'S', 'M', 'L'};
constexpr std::uint32_t version = 1;
constexpr std::uint32_t row_major = 1;

constexpr std::uint64_t max_elements = 100'000'000;
constexpr std::uint32_t max_string_length = 1'000'000;

template <typename T>
void write_value(std::ofstream& file, const T& value) {
    file.write(
        reinterpret_cast<const char*>(&value),
        sizeof(T)
    );

    if (!file) {
        throw std::runtime_error("Binary write failed");
    }
}

template <typename T>
T read_value(std::ifstream& file) {
    T value{};

    file.read(
        reinterpret_cast<char*>(&value),
        sizeof(T)
    );

    if (!file) {
        throw std::runtime_error("Binary read failed");
    }

    return value;
}

void write_string(
    std::ofstream& file,
    const std::string& text
) {
    if (text.size() > max_string_length) {
        throw std::runtime_error("String is too long");
    }

    write_value(
        file,
        static_cast<std::uint32_t>(text.size())
    );

    file.write(
        text.data(),
        static_cast<std::streamsize>(text.size())
    );

    if (!file) {
        throw std::runtime_error("String write failed");
    }
}

std::string read_string(std::ifstream& file) {
    const auto length = read_value<std::uint32_t>(file);

    if (length > max_string_length) {
        throw std::runtime_error(
            "Invalid string length in binary file"
        );
    }

    std::string text(length, '\0');

    file.read(
        text.data(),
        static_cast<std::streamsize>(length)
    );

    if (!file) {
        throw std::runtime_error("String read failed");
    }

    return text;
}

void validate_dimensions(
    std::uint64_t rows,
    std::uint64_t cols
) {
    if (cols == 0 || rows > max_elements ||
        cols > max_elements ||
        rows > max_elements / cols) {
        throw std::runtime_error(
            "Invalid binary matrix dimensions"
        );
    }
}

} // namespace

void save_binary_matrix(
    const NumericMatrix& matrix,
    const std::string& filename
) {
    validate_dimensions(matrix.rows, matrix.cols);

    if (matrix.values.size() != matrix.rows * matrix.cols ||
        matrix.column_names.size() != matrix.cols ||
        matrix.target_values.size() != matrix.rows) {
        throw std::runtime_error(
            "Inconsistent matrix dimensions"
        );
    }

    std::ofstream file(filename, std::ios::binary);

    if (!file) {
        throw std::runtime_error(
            "Cannot create binary file: " + filename
        );
    }

    file.write(magic, sizeof(magic));

    write_value(file, version);
    write_value(file, row_major);

    write_value(
        file,
        static_cast<std::uint64_t>(matrix.rows)
    );

    write_value(
        file,
        static_cast<std::uint64_t>(matrix.cols)
    );

    for (const auto& name : matrix.column_names) {
        write_string(file, name);
    }

    for (double value : matrix.values) {
        write_value(file, value);
    }

    for (const auto& target : matrix.target_values) {
        write_string(file, target);
    }

    file.flush();

    if (!file) {
        throw std::runtime_error(
            "Failed to finish writing binary file"
        );
    }
}

NumericMatrix load_binary_matrix(
    const std::string& filename
) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        throw std::runtime_error(
            "Cannot open binary file: " + filename
        );
    }

    char signature[4]{};
    file.read(signature, sizeof(signature));

    if (!file ||
        std::string(signature, sizeof(signature)) !=
            std::string(magic, sizeof(magic))) {
        throw std::runtime_error(
            "Invalid binary file signature"
        );
    }

    if (read_value<std::uint32_t>(file) != version) {
        throw std::runtime_error(
            "Unsupported binary file version"
        );
    }

    if (read_value<std::uint32_t>(file) != row_major) {
        throw std::runtime_error(
            "Unsupported matrix storage order"
        );
    }

    const auto rows = read_value<std::uint64_t>(file);
    const auto cols = read_value<std::uint64_t>(file);

    validate_dimensions(rows, cols);

    NumericMatrix matrix;
    matrix.rows = static_cast<std::size_t>(rows);
    matrix.cols = static_cast<std::size_t>(cols);

    matrix.column_names.reserve(matrix.cols);

    for (std::size_t i = 0; i < matrix.cols; ++i) {
        matrix.column_names.push_back(read_string(file));
    }

    matrix.values.resize(matrix.rows * matrix.cols);

    for (double& value : matrix.values) {
        value = read_value<double>(file);
    }

    matrix.target_values.reserve(matrix.rows);

    for (std::size_t i = 0; i < matrix.rows; ++i) {
        matrix.target_values.push_back(read_string(file));
    }

    return matrix;
}