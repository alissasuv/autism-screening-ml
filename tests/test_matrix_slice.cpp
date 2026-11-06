#include "preprocessing.hpp"

#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Function>
void check_throws(Function function, const char* message) {
    bool thrown = false;
    try {
        function();
    } catch (const std::exception&) {
        thrown = true;
    }
    check(thrown, message);
}

void test_layout(MatrixLayout layout) {
    NumericMatrix matrix;
    matrix.rows = 2;
    matrix.cols = 3;
    matrix.layout = layout;

    matrix.values =
        layout == MatrixLayout::RowMajor
            ? std::vector<double>{1, 2, 3, 4, 5, 6}
            : std::vector<double>{1, 4, 2, 5, 3, 6};

    check(matrix(0, 0) == 1, "Element (0,0)");
    check(matrix(1, 2) == 6, "Element (1,2)");

    auto row = matrix(0, std::nullopt);
    auto col = matrix(std::nullopt, 1);

    check(row.axis() == SliceAxis::Row, "Row axis");
    check(col.axis() == SliceAxis::Column, "Column axis");

    check(row.size() == 3, "Row size");
    check(col.size() == 2, "Column size");

    check(row(2) == 3, "Row indexing");
    check(col(1) == 5, "Column indexing");

    row(0) = 10;
    col(1) = 50;

    check(matrix(0, 0) == 10, "Row mutation");
    check(matrix(1, 1) == 50, "Column mutation");

    if (layout == MatrixLayout::RowMajor) {
        check(row.stride() == 1, "Row-major row stride");
        check(col.stride() == 3, "Row-major column stride");
        check(row.is_contiguous(), "Row should be contiguous");
        check(!col.is_contiguous(), "Column should be strided");

        check((*row)[0] == 10, "Row span");
        check_throws([&] { (void)*col; }, "Strided column span");
        check_throws([&] { (void)matrix[2]; }, "Row bounds");
    } else {
        check(row.stride() == 2, "Column-major row stride");
        check(col.stride() == 1, "Column-major column stride");
        check(!row.is_contiguous(), "Row should be strided");
        check(col.is_contiguous(), "Column should be contiguous");

        check((*col)[1] == 50, "Column span");
        check_throws([&] { (void)*row; }, "Strided row span");
        check_throws([&] { (void)matrix[0]; }, "Legacy row span");
    }

    check_throws(
        [&] { (void)row.at(row.size()); },
        "Slice bounds"
    );

    const NumericMatrix& readonly = matrix;
    auto const_row = readonly(0, std::nullopt);

    static_assert(std::is_same_v<
        decltype(const_row(0)), const double&
    >);

    check(const_row(0) == 10, "Const row");

    auto const_slice = row.as_const();
    check(const_slice(0) == 10, "as_const");

    static_assert(std::is_same_v<
        decltype(const_slice(0)), const double&
    >);
}

void test_empty_slices() {
    MatrixSlice<double> empty;
    check(empty.empty(), "Default slice must be empty");
    check(empty.data() == nullptr, "Empty slice pointer");
    check((*empty).empty(), "Empty span");

    NumericMatrix empty_row_matrix;
    empty_row_matrix.rows = 2;
    empty_row_matrix.cols = 0;

    auto row = empty_row_matrix(1, std::nullopt);
    check(row.empty(), "Empty matrix row");
    check(row.data() == nullptr, "Empty row pointer");

    NumericMatrix empty_col_matrix;
    empty_col_matrix.rows = 0;
    empty_col_matrix.cols = 2;

    auto col = empty_col_matrix(std::nullopt, 1);
    check(col.empty(), "Empty matrix column");
    check(col.data() == nullptr, "Empty column pointer");
}

void test_single_element() {
    double value = 7.0;
    MatrixSlice<double> slice(
        &value, 1, 5, SliceAxis::Column
    );

    check(slice.is_contiguous(), "Single element contiguous");
    check((*slice)[0] == 7.0, "Single element span");
}

} // namespace

int main() {
    try {
        test_layout(MatrixLayout::RowMajor);
        test_layout(MatrixLayout::ColumnMajor);
        test_empty_slices();
        test_single_element();

        std::cout << "[PASS] MatrixSlice tests\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
