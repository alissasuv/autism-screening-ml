#include "cart.hpp"
#include "csv_reader.hpp"
#include "dataset.hpp"
#include "feature_type.hpp"
#include "preprocessing.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

template <typename Model>
void print_structure(
    const Model& model,
    const NumericMatrix& features
) {
    std::cout << "Nodes: " << model.node_count()
              << ", leaves: " << model.leaf_count()
              << ", actual depth: " << model.depth()
              << '\n';

    for (const auto& split : model.splits(5)) {
        const std::string name =
            split.feature < features.column_names.size()
                ? features.column_names[split.feature]
                : "feature " + std::to_string(split.feature);

        std::cout << "  depth " << split.depth
                  << ": " << name
                  << " <= " << split.threshold << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string path =
            argc > 1
                ? argv[1]
                : "data/Autism-Adult-Data.csv";

        const auto raw = read_csv(path);

        std::vector<FeatureType> types;
        types.reserve(raw.columns.size());

        for (const auto& column : raw.columns) {
            types.push_back(
                detect_feature_type(column)
            );
        }

        const auto data = make_typed_dataset(raw, types);
        const auto split = split_dataset(data, 0.8, 42);

        const auto state = fit_preprocessing(
            split.train,
            PreprocessingOptions{}
        );

        const auto train = transform_preprocessing(
            split.train,
            state
        );

        const auto test = transform_preprocessing(
            split.test,
            state
        );

        const auto y_train = encode_target(
            train.target_values
        );

        const auto y_test = encode_target(
            test.target_values
        );

        std::cout
            << "CART classification (Autism Screening Adult)\n"
            << "Train: " << train.rows
            << ", test: " << test.rows
            << ", features: " << train.cols << '\n';

        for (std::size_t depth : {1U, 4U}) {
            CartClassifier classifier({depth, 2, 5});

            IClassifier& model = classifier;
            model.fit(train, y_train);

            const auto predictions = model.predict(test);
            const auto proba = model.predict_proba(test);
            const auto classes = model.classes();

            std::cout << "\nmax_depth=" << depth << '\n';
            print_structure(classifier, train);

            const std::size_t shown =
                std::min<std::size_t>(5, test.rows);

            for (std::size_t i = 0; i < shown; ++i) {
                std::cout
                    << "Row " << i
                    << ": predicted=" << predictions[i]
                    << ", actual=" << y_test[i]
                    << ", probabilities=[";

                for (std::size_t j = 0;
                     j < classes.size();
                     ++j) {
                    if (j != 0) {
                        std::cout << ", ";
                    }

                    std::cout
                        << classes[j] << ":"
                        << std::fixed
                        << std::setprecision(3)
                        << proba(i, j);
                }

                std::cout << "]\n";
            }
        }

        // Independent numerical regression example.
        // Class labels are not treated as regression targets.

        NumericMatrix regression_x;
        regression_x.rows = 8;
        regression_x.cols = 1;
        regression_x.column_names = {"x"};
        regression_x.values = {
            1, 2, 3, 4, 5, 6, 7, 8
        };

        const std::vector<double> regression_y = {
            2, 2, 3, 3, 10, 10, 12, 12
        };

        CartRegressor regressor({3, 2, 1});

        IRegressor& regression_model = regressor;

        regression_model.fit(
            regression_x,
            regression_y
        );

        const auto regression_predictions =
            regression_model.predict(regression_x);

        std::cout
            << "\nCART regression (synthetic data)\n";

        print_structure(regressor, regression_x);

        for (std::size_t i = 0;
             i < regression_x.rows;
             ++i) {
            std::cout
                << "x=" << regression_x(i, 0)
                << ", predicted=" << regression_predictions[i]
                << ", actual=" << regression_y[i]
                << '\n';
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "Error: " << error.what() << '\n';

        return 1;
    }
}







