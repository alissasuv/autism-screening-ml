#include "split_finder.hpp"
#include "ml_exceptions.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

NumericMatrix matrix(
    std::size_t rows,
    std::size_t cols,
    std::vector<double> values,
    MatrixLayout layout = MatrixLayout::RowMajor
) {
    NumericMatrix result;

    result.rows = rows;
    result.cols = cols;
    result.layout = layout;
    result.values.resize(values.size());

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            result.values[
                layout == MatrixLayout::RowMajor
                    ? i * cols + j
                    : j * rows + i
            ] = values[i * cols + j];
        }
    }

    return result;
}

void run_tests(MatrixLayout layout) {
    SplitFinder finder;

    const auto X = matrix(
        4,
        2,
        {
            0, 5,
            1, 5,
            2, 5,
            3, 5
        },
        layout
    );

    const std::vector<double> y = {0, 0, 1, 1};

    const std::vector<std::size_t> ids = {
        0, 1, 2, 3
    };

    const std::vector<std::size_t> features = {
        0, 1
    };

    const std::vector<std::size_t> counts = {
        2, 1, 3, 1
    };

    const auto unweighted = finder.find_best(
        X,
        y,
        ids,
        features,
        SplitCriterion::Gini,
        1
    );

    require(
        unweighted &&
        unweighted->feature == 0 &&
        unweighted->threshold == 1.5,
        "Unweighted Gini split"
    );

    const auto weighted = finder.find_best(
        X,
        y,
        ids,
        features,
        SplitCriterion::Gini,
        1,
        counts
    );

    require(
        weighted &&
        weighted->feature == 0 &&
        weighted->threshold == 1.5,
        "Weighted Gini split"
    );

    require(
        std::abs(weighted->score - 7.0) < 1e-10,
        "Weighted Gini score must use multiplicities"
    );

    const auto regression = finder.find_for_feature(
        X,
        y,
        ids,
        0,
        SplitCriterion::Variance,
        1,
        counts
    );

    require(
        !regression.is_constant &&
        regression.best &&
        regression.best->threshold == 1.5,
        "Weighted regression split"
    );

    require(
        std::abs(regression.best->score - 4.0) < 1e-10,
        "Weighted regression score"
    );

    const auto constant = finder.find_for_feature(
        X,
        y,
        ids,
        1,
        SplitCriterion::Gini,
        1,
        counts
    );

    require(
        constant.is_constant && !constant.best,
        "Constant feature"
    );

    const auto impossible = finder.find_for_feature(
        X,
        y,
        ids,
        0,
        SplitCriterion::Gini,
        3,
        counts
    );

    require(
        !impossible.is_constant && !impossible.best,
        "Leaf minimum uses unique object indices"
    );

    const std::vector<std::size_t> duplicate_ids = {
        0, 0, 1
    };

    bool rejected = false;

    try {
        (void)finder.find_for_feature(
            X,
            y,
            duplicate_ids,
            0,
            SplitCriterion::Gini
        );
    } catch (const InvalidModelParameterError&) {
        rejected = true;
    }

    require(
        rejected,
        "Repeated object IDs must be rejected"
    );

    const std::vector<std::size_t> invalid_counts = {
        1, 1
    };

    rejected = false;

    try {
        (void)finder.find_for_feature(
            X,
            y,
            ids,
            0,
            SplitCriterion::Gini,
            1,
            invalid_counts
        );
    } catch (const DimensionMismatchError&) {
        rejected = true;
    }

    require(
        rejected,
        "Invalid count dimensions must be rejected"
    );
}

} // namespace

int main() {
    try {
        run_tests(MatrixLayout::RowMajor);
        run_tests(MatrixLayout::ColumnMajor);

        std::cout << "SplitFinder tests passed\n";
    } catch (const std::exception& error) {
        std::cerr
            << "SplitFinder tests failed: "
            << error.what()
            << '\n';

        return 1;
    }

    return 0;
}




