#include "lab5_data.hpp"

#include "feature_type.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t file_version = 1;
constexpr std::uint64_t max_rows = 10'000'000;
constexpr std::uint64_t max_columns = 100'000;
constexpr std::uint64_t max_cells = 100'000'000;
constexpr std::uint64_t max_string_size = 1'000'000;

void validate_raw(const RawDataset& data) {
    if (data.row_count == 0 ||
        data.columns.empty() ||
        data.columns.size() != data.column_names.size()) {
        throw std::invalid_argument("Invalid raw dataset dimensions");
    }

    if (data.row_count > max_rows ||
        data.columns.size() > max_columns ||
        data.columns.size() > max_cells / data.row_count) {
        throw std::invalid_argument("Raw dataset is too large");
    }

    for (const auto& column : data.columns) {
        if (column.size() != data.row_count) {
            throw std::invalid_argument(
                "Raw dataset contains inconsistent column lengths"
            );
        }
    }
}

std::size_t find_column(
    const RawDataset& data,
    const std::string& name
) {
    const auto it = std::find(
        data.column_names.begin(),
        data.column_names.end(),
        name
    );

    if (it == data.column_names.end()) {
        throw std::invalid_argument(
            "Column not found: " + name
        );
    }

    return static_cast<std::size_t>(
        std::distance(data.column_names.begin(), it)
    );
}

PreparedDataset select_rows(
    const RawDataset& source,
    const std::vector<std::size_t>& indices
) {
    PreparedDataset result;

    result.data.column_names = source.column_names;
    result.data.row_count = indices.size();
    result.data.columns.resize(source.columns.size());

    result.source_indices.reserve(indices.size());

    for (const std::size_t index : indices) {
        if (index >= source.row_count) {
            throw std::out_of_range("Source row index out of range");
        }

        result.source_indices.push_back(
            static_cast<std::uint64_t>(index)
        );
    }

    for (std::size_t col = 0; col < source.columns.size(); ++col) {
        auto& destination = result.data.columns[col];
        destination.reserve(indices.size());

        for (const std::size_t index : indices) {
            destination.push_back(source.columns[col][index]);
        }
    }

    return result;
}

template <typename T>
void write_number(std::ostream& out, T value) {
    out.write(
        reinterpret_cast<const char*>(&value),
        sizeof(value)
    );

    if (!out) {
        throw std::runtime_error("Failed to write binary value");
    }
}

template <typename T>
T read_number(std::istream& in) {
    T value{};

    in.read(
        reinterpret_cast<char*>(&value),
        sizeof(value)
    );

    if (!in) {
        throw std::runtime_error("Failed to read binary value");
    }

    return value;
}

void write_string(std::ostream& out, const std::string& value) {
    if (value.size() > max_string_size) {
        throw std::runtime_error("String exceeds binary format limit");
    }

    write_number<std::uint64_t>(
        out,
        static_cast<std::uint64_t>(value.size())
    );

    out.write(
        value.data(),
        static_cast<std::streamsize>(value.size())
    );

    if (!out) {
        throw std::runtime_error("Failed to write string");
    }
}

std::string read_string(std::istream& in) {
    const auto size = read_number<std::uint64_t>(in);

    if (size > max_string_size) {
        throw std::runtime_error("Invalid binary string size");
    }

    std::string result(static_cast<std::size_t>(size), '\0');

    in.read(
        result.data(),
        static_cast<std::streamsize>(size)
    );

    if (!in) {
        throw std::runtime_error("Failed to read string");
    }

    return result;
}

std::string json_string(const std::string& value) {
    std::ostringstream out;
    out << '"';

    constexpr char hex[] = "0123456789abcdef";

    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u00"
                        << hex[ch >> 4]
                        << hex[ch & 0x0f];
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }

    out << '"';
    return out.str();
}

std::map<std::string, std::size_t> class_counts(
    const RawDataset& data,
    const std::string& target_name
) {
    const auto target_index = find_column(data, target_name);
    std::map<std::string, std::size_t> counts;

    for (const auto& cell : data.columns[target_index]) {
        if (!cell.has_value()) {
            throw std::invalid_argument(
                "Target contains missing values"
            );
        }

        ++counts[*cell];
    }

    return counts;
}

void write_class_counts(
    std::ostream& out,
    const std::map<std::string, std::size_t>& counts
) {
    out << "{";

    bool first = true;

    for (const auto& [label, count] : counts) {
        if (!first) {
            out << ", ";
        }

        first = false;

        out << json_string(label) << ": " << count;
    }

    out << "}";
}

void write_column_statistics(
    std::ostream& out,
    const RawDataset& data
) {
    out << "[\n";

    for (std::size_t col = 0; col < data.columns.size(); ++col) {
        if (col != 0) {
            out << ",\n";
        }

        const auto& values = data.columns[col];

        std::size_t missing = 0;
        std::map<std::string, std::size_t> frequencies;

        for (const auto& cell : values) {
            if (!cell.has_value()) {
                ++missing;
            } else {
                ++frequencies[*cell];
            }
        }

        const FeatureType type = detect_feature_type(values);

        out << "    {\n";
        out << "      \"name\": "
            << json_string(data.column_names[col]) << ",\n";
        out << "      \"type\": "
            << json_string(
                type == FeatureType::Numerical
                    ? "numeric"
                    : "categorical"
            ) << ",\n";
        out << "      \"missing\": " << missing << ",\n";
        out << "      \"observed\": "
            << (values.size() - missing) << ",\n";
        out << "      \"unique\": " << frequencies.size();

        if (type == FeatureType::Numerical) {
            std::vector<double> numbers;
            numbers.reserve(values.size() - missing);

            for (const auto& cell : values) {
                if (cell.has_value()) {
                    numbers.push_back(std::stod(*cell));
                }
            }

            if (!numbers.empty()) {
                const auto [lo, hi] = std::minmax_element(
                    numbers.begin(),
                    numbers.end()
                );

                const double sum = std::accumulate(
                    numbers.begin(), numbers.end(), 0.0
                );

                out << ",\n      \"min\": " << *lo;
                out << ",\n      \"max\": " << *hi;
                out << ",\n      \"mean\": "
                    << (sum / static_cast<double>(numbers.size()));
            }
        } else {
            out << ",\n      \"frequencies\": {";

            bool first = true;

            for (const auto& [value, count] : frequencies) {
                if (!first) {
                    out << ", ";
                }

                first = false;

                out << json_string(value) << ": " << count;
            }

            out << "}";
        }

        out << "\n    }";
    }

    out << "\n  ]";
}

} // namespace

PreparedSplit make_stratified_split(
    const RawDataset& source,
    const std::string& target_name,
    double test_fraction,
    std::uint32_t seed
) {
    validate_raw(source);

    if (!std::isfinite(test_fraction) ||
        test_fraction <= 0.0 ||
        test_fraction >= 1.0) {
        throw std::invalid_argument(
            "test_fraction must be between 0 and 1"
        );
    }

    const auto target_index = find_column(source, target_name);

    std::map<std::string, std::vector<std::size_t>> groups;

    for (std::size_t row = 0; row < source.row_count; ++row) {
        const auto& label = source.columns[target_index][row];

        if (!label.has_value()) {
            throw std::invalid_argument(
                "Classification target contains missing values"
            );
        }

        groups[*label].push_back(row);
    }

    if (groups.size() < 2) {
        throw std::invalid_argument(
            "Stratified split requires at least two classes"
        );
    }

    const auto requested_test = static_cast<std::size_t>(
        std::llround(
            static_cast<double>(source.row_count) * test_fraction
        )
    );

    if (requested_test < groups.size() ||
        source.row_count - requested_test < groups.size()) {
        throw std::invalid_argument(
            "Train/test sizes are too small for stratification"
        );
    }

    struct GroupAllocation {
        std::vector<std::size_t> rows;
        std::size_t test_count = 0;
        double ideal_test_count = 0.0;
    };

    std::vector<GroupAllocation> allocations;
    allocations.reserve(groups.size());

    std::mt19937 rng(seed);
    std::size_t allocated_test = 0;

    for (auto& [label, rows] : groups) {
        (void)label;

        if (rows.size() < 2) {
            throw std::invalid_argument(
                "Every class must contain at least two objects"
            );
        }

        std::shuffle(rows.begin(), rows.end(), rng);

        const double ideal =
            static_cast<double>(rows.size()) * test_fraction;

        const auto rounded = static_cast<std::size_t>(
            std::llround(ideal)
        );

        const std::size_t count = std::clamp(
            rounded,
            std::size_t{1},
            rows.size() - 1
        );

        allocated_test += count;

        allocations.push_back({
            std::move(rows),
            count,
            ideal
        });
    }

    // Adjust allocations to the requested total test size.
    while (allocated_test != requested_test) {
        const bool increase = allocated_test < requested_test;

        std::optional<std::size_t> best;
        double best_priority =
            -std::numeric_limits<double>::infinity();

        for (std::size_t i = 0; i < allocations.size(); ++i) {
            const auto& group = allocations[i];

            const bool eligible = increase
                ? group.test_count < group.rows.size() - 1
                : group.test_count > 1;

            if (!eligible) {
                continue;
            }

            const double priority = increase
                ? group.ideal_test_count -
                    static_cast<double>(group.test_count)
                : static_cast<double>(group.test_count) -
                    group.ideal_test_count;

            if (!best.has_value() ||
                priority > best_priority) {
                best = i;
                best_priority = priority;
            }
        }

        if (!best.has_value()) {
            throw std::invalid_argument(
                "Cannot satisfy requested stratified split size"
            );
        }

        if (increase) {
            ++allocations[*best].test_count;
            ++allocated_test;
        } else {
            --allocations[*best].test_count;
            --allocated_test;
        }
    }

    std::vector<std::size_t> train_indices;
    std::vector<std::size_t> test_indices;

    for (const auto& group : allocations) {
        for (std::size_t i = 0; i < group.rows.size(); ++i) {
            if (i < group.test_count) {
                test_indices.push_back(group.rows[i]);
            } else {
                train_indices.push_back(group.rows[i]);
            }
        }
    }

    std::shuffle(train_indices.begin(), train_indices.end(), rng);
    std::shuffle(test_indices.begin(), test_indices.end(), rng);

    PreparedSplit result;
    result.train = select_rows(source, train_indices);
    result.test = select_rows(source, test_indices);

    return result;
}

void save_prepared_dataset(
    const PreparedDataset& dataset,
    const std::string& filename
) {
    validate_raw(dataset.data);

    if (dataset.source_indices.size() != dataset.data.row_count) {
        throw std::invalid_argument(
            "Source indices do not match dataset size"
        );
    }

    std::ofstream out(filename, std::ios::binary);

    if (!out) {
        throw std::runtime_error(
            "Cannot create dataset file: " + filename
        );
    }

    out.write("RAW5", 4);
    write_number<std::uint32_t>(out, file_version);

    write_number<std::uint64_t>(
        out, static_cast<std::uint64_t>(dataset.data.row_count)
    );

    write_number<std::uint64_t>(
        out, static_cast<std::uint64_t>(dataset.data.columns.size())
    );

    for (const auto& name : dataset.data.column_names) {
        write_string(out, name);
    }

    for (std::uint64_t index : dataset.source_indices) {
        write_number<std::uint64_t>(out, index);
    }

    for (const auto& column : dataset.data.columns) {
        for (const auto& cell : column) {
            write_number<std::uint8_t>(
                out,
                static_cast<std::uint8_t>(cell.has_value())
            );

            if (cell.has_value()) {
                write_string(out, *cell);
            }
        }
    }

    out.flush();

    if (!out) {
        throw std::runtime_error(
            "Failed to finish writing dataset file"
        );
    }
}

PreparedDataset load_prepared_dataset(
    const std::string& filename
) {
    std::ifstream in(filename, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "Cannot open dataset file: " + filename
        );
    }

    char magic[4]{};
    in.read(magic, 4);

    if (!in || std::string(magic, 4) != "RAW5") {
        throw std::runtime_error(
            "Invalid raw dataset file signature"
        );
    }

    if (read_number<std::uint32_t>(in) != file_version) {
        throw std::runtime_error(
            "Unsupported raw dataset file version"
        );
    }

    const std::uint64_t rows = read_number<std::uint64_t>(in);
    const std::uint64_t cols = read_number<std::uint64_t>(in);

    if (rows == 0 || cols == 0 ||
        rows > max_rows ||
        cols > max_columns ||
        cols > max_cells / rows) {
        throw std::runtime_error(
            "Invalid raw dataset file dimensions"
        );
    }

    PreparedDataset result;
    result.data.row_count = static_cast<std::size_t>(rows);
    result.data.column_names.reserve(static_cast<std::size_t>(cols));
    result.data.columns.resize(static_cast<std::size_t>(cols));
    result.source_indices.reserve(static_cast<std::size_t>(rows));

    for (std::uint64_t col = 0; col < cols; ++col) {
        result.data.column_names.push_back(read_string(in));
    }

    for (std::uint64_t row = 0; row < rows; ++row) {
        result.source_indices.push_back(
            read_number<std::uint64_t>(in)
        );
    }

    for (auto& column : result.data.columns) {
        column.reserve(static_cast<std::size_t>(rows));

        for (std::uint64_t row = 0; row < rows; ++row) {
            const auto present = read_number<std::uint8_t>(in);

            if (present == 0) {
                column.push_back(std::nullopt);
            } else if (present == 1) {
                column.push_back(read_string(in));
            } else {
                throw std::runtime_error(
                    "Invalid missing-value marker"
                );
            }
        }
    }

    validate_raw(result.data);
    return result;
}

void save_dataset_statistics(
    const RawDataset& source,
    const PreparedSplit& split,
    const std::string& target_name,
    double test_fraction,
    std::uint32_t seed,
    const std::string& filename
) {
    validate_raw(source);
    validate_raw(split.train.data);
    validate_raw(split.test.data);

    std::ofstream out(filename);

    if (!out) {
        throw std::runtime_error(
            "Cannot create statistics file: " + filename
        );
    }

    out << std::setprecision(17);

    out << "{\n";
    out << "  \"target\": " << json_string(target_name) << ",\n";
    out << "  \"task\": \"classification\",\n";
    out << "  \"random_state\": " << seed << ",\n";
    out << "  \"test_fraction\": " << test_fraction << ",\n";
    out << "  \"total_rows\": " << source.row_count << ",\n";
    out << "  \"train_rows\": " << split.train.data.row_count << ",\n";
    out << "  \"test_rows\": " << split.test.data.row_count << ",\n";

    out << "  \"class_distribution\": {\n";

    out << "    \"all\": ";
    write_class_counts(out, class_counts(source, target_name));
    out << ",\n";

    out << "    \"train\": ";
    write_class_counts(
        out, class_counts(split.train.data, target_name)
    );
    out << ",\n";

    out << "    \"test\": ";
    write_class_counts(
        out, class_counts(split.test.data, target_name)
    );
    out << "\n  },\n";

    out << "  \"columns\": {\n";

    out << "    \"all\": ";
    write_column_statistics(out, source);
    out << ",\n";

    out << "    \"train\": ";
    write_column_statistics(out, split.train.data);
    out << ",\n";

    out << "    \"test\": ";
    write_column_statistics(out, split.test.data);
    out << "\n  }\n";

    out << "}\n";

    if (!out) {
        throw std::runtime_error(
            "Failed to write dataset statistics"
        );
    }
}

