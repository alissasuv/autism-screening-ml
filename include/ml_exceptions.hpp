#pragma once

#include <stdexcept>
#include <string>

class DimensionMismatchError : public std::invalid_argument {
public:
    explicit DimensionMismatchError(const std::string& message)
        : std::invalid_argument(message) {}
};

class InvalidModelStateError : public std::runtime_error {
public:
    explicit InvalidModelStateError(const std::string& message)
        : std::runtime_error(message) {}
};

class InvalidModelParameterError : public std::invalid_argument {
public:
    explicit InvalidModelParameterError(const std::string& message)
        : std::invalid_argument(message) {}
};
