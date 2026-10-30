
#include "decision_stump.hpp"
#include "ml_exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool close(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) <= eps *
        std::max({1.0, std::abs(a), std::abs(b)});
}

NumericMatrix make_matrix(
    const std::vector<std::vector<double>>& rows
) {
    NumericMatrix matrix;
    matrix.rows = rows.size();
    matrix.cols = rows.empty() ? 0 : rows.front().size();

    for (const auto& row : rows) {
        require(row.size() == matrix.cols, "Invalid test matrix");
        matrix.values.insert(
            matrix.values.end(), row.begin(), row.end()
        );
    }

    return matrix;
}

template <typename Exception, typename Function>
void expect_throw(Function function) {
    bool caught = false;

    try {
        function();
    } catch (const Exception&) {
        caught = true;
    }

    require(caught, "Expected exception was not thrown");
}

struct ReferenceSplit {
    bool found = false;
    std::size_t feature = 0;
    double threshold = 0.0;
    double score = -std::numeric_limits<double>::infinity();
};

double reference_score(
    const std::vector<double>& left,
    const std::vector<double>& right,
    SplitCriterion criterion
) {
    if (criterion == SplitCriterion::Variance) {
        const double left_sum =
            std::accumulate(left.begin(), left.end(), 0.0);
        const double right_sum =
            std::accumulate(right.begin(), right.end(), 0.0);

        return left_sum * left_sum / left.size() +
               right_sum * right_sum / right.size();
    }

    auto score_part = [](const std::vector<double>& values) {
        std::map<double, std::size_t> counts;

        for (double value : values) {
            ++counts[value];
        }

        double squared_sum = 0.0;

        for (const auto& [label, count] : counts) {
            (void)label;
            squared_sum += static_cast<double>(count) * count;
        }

        return squared_sum / values.size();
    };

    return score_part(left) + score_part(right);
}

// Intentionally slow reference implementation.
// It recalculates each candidate split from scratch.
ReferenceSplit naive_split(
    const NumericMatrix& x,
    const std::vector<double>& y,
    SplitCriterion criterion,
    std::size_t min_leaf
) {
    ReferenceSplit best;

    for (std::size_t j = 0; j < x.cols; ++j) {
        std::vector<double> values;

        for (std::size_t i = 0; i < x.rows; ++i) {
            values.push_back(x[i][j]);
        }

        std::sort(values.begin(), values.end());
        values.erase(
            std::unique(values.begin(), values.end()),
            values.end()
        );

        for (std::size_t p = 0; p + 1 < values.size(); ++p) {
            const double threshold =
                std::midpoint(values[p], values[p + 1]);

            std::vector<double> left;
            std::vector<double> right;

            for (std::size_t i = 0; i < x.rows; ++i) {
                if (x[i][j] <= threshold) {
                    left.push_back(y[i]);
                } else {
                    right.push_back(y[i]);
                }
            }

            if (left.size() < min_leaf ||
                right.size() < min_leaf) {
                continue;
            }

            const double score =
                reference_score(left, right, criterion);

            if (!best.found || score > best.score) {
                best.found = true;
                best.feature = j;
                best.threshold = threshold;
                best.score = score;
            }
        }
    }

    return best;
}

void test_regression() {
    const auto x = make_matrix({
        {0}, {1}, {2}, {3}
    });

    const std::vector<double> y = {2, 2, 8, 8};

    DecisionStump stump(SplitCriterion::Variance);
    stump.fit(x, y);

    require(stump.has_split(), "Regression split missing");
    require(stump.split_feature() == 0, "Wrong feature");
    require(close(stump.split_threshold(), 1.5),
            "Wrong regression threshold");

    const auto predictions = stump.predict(x);

    for (std::size_t i = 0; i < y.size(); ++i) {
        require(close(predictions[i], y[i]),
                "Wrong regression prediction");
    }
}

void test_classification() {
    const auto x = make_matrix({
        {0}, {1}, {2}, {3}
    });

    const std::vector<double> y = {0, 0, 1, 1};

    DecisionStump stump(SplitCriterion::Gini);
    IModel& model = stump;

    model.fit(x, y);
    const auto predictions = model.predict(x);

    require(stump.has_split(), "Classification split missing");
    require(close(stump.split_threshold(), 1.5),
            "Wrong classification threshold");
    require(predictions == y, "Wrong classification predictions");
}

void test_constant_features() {
    const auto x = make_matrix({
        {5, 5}, {5, 5}, {5, 5}, {5, 5}
    });

    const std::vector<double> y = {0, 1, 1, 1};

    DecisionStump stump(SplitCriterion::Gini);
    stump.fit(x, y);

    require(!stump.has_split(), "Unexpected split");

    for (double value : stump.predict(x)) {
        require(value == 1, "Wrong constant prediction");
    }

    expect_throw<InvalidModelStateError>([&] {
        (void)stump.split_threshold();
    });
}

void test_one_object_and_one_class() {
    const auto single = make_matrix({{3}});
    const std::vector<double> single_y = {7};

    DecisionStump regression(SplitCriterion::Variance);
    regression.fit(single, single_y);

    require(!regression.has_split(), "Single object was split");
    require(close(regression.predict(single)[0], 7),
            "Wrong single-object prediction");

    const auto x = make_matrix({{0}, {1}, {2}});
    const std::vector<double> y = {1, 1, 1};

    DecisionStump classifier(SplitCriterion::Gini);
    classifier.fit(x, y);

    for (double value : classifier.predict(x)) {
        require(value == 1, "Wrong single-class prediction");
    }
}

void test_against_naive() {
    std::mt19937 generator(42);
    std::uniform_int_distribution<int> feature_dist(0, 20);
    std::uniform_int_distribution<int> class_dist(0, 2);

    for (SplitCriterion criterion :
         {SplitCriterion::Variance, SplitCriterion::Gini}) {

        for (std::size_t min_leaf : {1u, 2u, 5u, 25u}) {
            for (int trial = 0; trial < 30; ++trial) {
                NumericMatrix x;
                x.rows = 30;
                x.cols = 3;

                std::vector<double> y;

                for (std::size_t i = 0; i < x.rows; ++i) {
                    for (std::size_t j = 0; j < x.cols; ++j) {
                        x.values.push_back(feature_dist(generator));
                    }

                    y.push_back(
                        criterion == SplitCriterion::Gini
                            ? class_dist(generator)
                            : feature_dist(generator)
                    );
                }

                DecisionStump stump(criterion, min_leaf);
                stump.fit(x, y);

                const auto expected =
                    naive_split(x, y, criterion, min_leaf);

                require(stump.has_split() == expected.found,
                        "Split existence differs from reference");

                if (expected.found) {
                    require(
                        stump.split_feature() == expected.feature,
                        "Split feature differs from reference"
                    );

                    require(
                        close(stump.split_threshold(),
                              expected.threshold),
                        "Split threshold differs from reference"
                    );
                }
            }
        }
    }
}

void test_exceptions() {
    expect_throw<InvalidModelParameterError>([] {
        DecisionStump stump(SplitCriterion::Gini, 0);
    });

    DecisionStump stump(SplitCriterion::Gini);

    const auto x = make_matrix({{0}, {1}});
    const std::vector<double> y = {0, 1};

    expect_throw<InvalidModelStateError>([&] {
        (void)stump.predict(x);
    });

    expect_throw<DimensionMismatchError>([&] {
        stump.fit(x, std::vector<double>{0});
    });

    expect_throw<InvalidModelParameterError>([&] {
        NumericMatrix empty;
        stump.fit(empty, std::vector<double>{});
    });

    stump.fit(x, y);

    expect_throw<DimensionMismatchError>([&] {
        const auto wrong = make_matrix({{1, 2}});
        (void)stump.predict(wrong);
    });

    expect_throw<std::out_of_range>([&] {
        (void)x.at(10);
    });
}

int main() {
    try {
        test_regression();
        test_classification();
        test_constant_features();
        test_one_object_and_one_class();
        test_against_naive();
        test_exceptions();

        std::cout << "All decision stump tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Decision stump test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
