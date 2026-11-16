#include "lab5_data.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

RawDataset make_test_dataset() {
    RawDataset data;

    data.column_names = {
        "age",
        "city",
        "target"
    };

    data.row_count = 10;

    data.columns = {
        {
            "20", "21", "22", "23", "24",
            "25", "26", "27", "28", "29"
        },
        {
            "London", "Paris", std::nullopt, "Berlin", "Rome",
            "Madrid", "London", "Paris", "Berlin", "Rome"
        },
        {
            "NO", "NO", "NO", "NO", "NO",
            "YES", "YES", "YES", "YES", "YES"
        }
    };

    return data;
}

std::size_t count_class(
    const PreparedDataset& dataset,
    const std::string& label
) {
    const auto& target = dataset.data.columns[2];

    return static_cast<std::size_t>(
        std::count_if(
            target.begin(),
            target.end(),
            [&](const auto& value) {
                return value.has_value() && *value == label;
            }
        )
    );
}

void test_split_sizes_and_stratification() {
    const RawDataset data = make_test_dataset();

    const PreparedSplit split = make_stratified_split(
        data, "target", 0.2, 42
    );

    assert(split.train.data.row_count == 8);
    assert(split.test.data.row_count == 2);

    assert(count_class(split.train, "NO") == 4);
    assert(count_class(split.train, "YES") == 4);

    assert(count_class(split.test, "NO") == 1);
    assert(count_class(split.test, "YES") == 1);

    std::cout << "[PASS] Split sizes and stratification\n";
}

void test_no_overlap_and_full_coverage() {
    const RawDataset data = make_test_dataset();

    const PreparedSplit split = make_stratified_split(
        data, "target", 0.2, 42
    );

    std::set<std::uint64_t> indices;

    for (const auto index : split.train.source_indices) {
        assert(indices.insert(index).second);
    }

    for (const auto index : split.test.source_indices) {
        assert(indices.insert(index).second);
    }

    assert(indices.size() == data.row_count);

    for (std::uint64_t i = 0; i < data.row_count; ++i) {
        assert(indices.contains(i));
    }

    std::cout << "[PASS] No overlap and full coverage\n";
}

void test_reproducibility() {
    const RawDataset data = make_test_dataset();

    const PreparedSplit first = make_stratified_split(
        data, "target", 0.2, 42
    );

    const PreparedSplit second = make_stratified_split(
        data, "target", 0.2, 42
    );

    assert(
        first.train.source_indices ==
        second.train.source_indices
    );

    assert(
        first.test.source_indices ==
        second.test.source_indices
    );

    assert(first.train.data.columns == second.train.data.columns);
    assert(first.test.data.columns == second.test.data.columns);

    std::cout << "[PASS] Reproducible split\n";
}

void test_binary_roundtrip() {
    const RawDataset data = make_test_dataset();

    const PreparedSplit split = make_stratified_split(
        data, "target", 0.2, 42
    );

    const auto filename =
        std::filesystem::temp_directory_path() /
        "lab5_data_roundtrip_test.bin";

    try {
        save_prepared_dataset(
            split.train,
            filename.string()
        );

        const PreparedDataset restored =
            load_prepared_dataset(filename.string());

        assert(
            restored.data.column_names ==
            split.train.data.column_names
        );

        assert(
            restored.data.columns ==
            split.train.data.columns
        );

        assert(
            restored.data.row_count ==
            split.train.data.row_count
        );

        assert(
            restored.source_indices ==
            split.train.source_indices
        );

        std::filesystem::remove(filename);
    } catch (...) {
        std::filesystem::remove(filename);
        throw;
    }

    std::cout << "[PASS] Binary round-trip\n";
}

void test_invalid_arguments() {
    const RawDataset data = make_test_dataset();

    bool caught = false;

    try {
        (void)make_stratified_split(
            data, "target", 1.0, 42
        );
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    assert(caught);

    caught = false;

    try {
        (void)make_stratified_split(
            data, "unknown_target", 0.2, 42
        );
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    assert(caught);

    std::cout << "[PASS] Invalid arguments\n";
}

} // namespace

int main() {
    try {
        test_split_sizes_and_stratification();
        test_no_overlap_and_full_coverage();
        test_reproducibility();
        test_binary_roundtrip();
        test_invalid_arguments();

        std::cout << "All Lab 5 data tests passed.\n";
        return 0;

    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}



