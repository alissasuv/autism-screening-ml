# Autism Screening Adult Machine Learning Project

## Project Overview
This project implements a C++20 data processing pipeline for the
UCI Autism Screening Adult dataset. Laboratory Work No. 1 focuses
on CSV loading, exploratory statistics, preprocessing, categorical
encoding, and binary serialization. The resulting numerical dataset
will be used in subsequent machine learning laboratory works.

## Dataset Description
- Dataset: Autism Screening Adult (UCI)
- Source: https://archive.ics.uci.edu/dataset/426/autism+screening+adult
- Records: 704
- Features: 20
- Target: Class/ASD (screening classification, not confirmed diagnosis)

## Feature Table
| Column | Type | Description |
|--------|------|-------------|
| A1_Score | Categorical | Binary AQ-10 questionnaire score |
| A2_Score | Categorical | Binary AQ-10 questionnaire score |
| A3_Score | Categorical | Binary AQ-10 questionnaire score |
| A4_Score | Categorical | Binary AQ-10 questionnaire score |
| A5_Score | Categorical | Binary AQ-10 questionnaire score |
| A6_Score | Categorical | Binary AQ-10 questionnaire score |
| A7_Score | Categorical | Binary AQ-10 questionnaire score |
| A8_Score | Categorical | Binary AQ-10 questionnaire score |
| A9_Score | Categorical | Binary AQ-10 questionnaire score |
| A10_Score | Categorical | Binary AQ-10 questionnaire score |
| age | Numerical | Age of the individual |
| gender | Categorical | Gender of the individual |
| ethnicity | Categorical | Ethnicity of the individual |
| jundice | Categorical | Presence of jaundice |
| austim | Categorical | Family history of autism |
| contry_of_res | Categorical | Country of residence |
| used_app_before | Categorical | Previous app usage |
| result | Numerical | Total AQ-10 screening score |
| age_desc | Categorical | Age description |
| relation | Categorical | Relationship |
| Class/ASD | Target | Screening classification |

## Missing Values
- age: 2 missing values
- ethnicity: 95 missing values
- relation: 95 missing values

## Feature Type Detection Rule
- Non-numeric features are categorical
- Numeric features with at most 2 unique values are categorical
- All other numeric features are numerical
- Missing values are represented by '?' and are ignored during type detection

## Build and Run
- **Requirements**: C++20, CMake, Clang/GCC
- **Build**: `cmake -S . -B build && cmake --build build`
- **Run**: `./build/autism_screening`
- **Help**: `./build/autism_screening --help`

## CLI Options
- `--input` (default: `data/Autism-Adult-Data.csv`)
- `--output` (default: `data/processed.bin`)
- `--bins` (default: 10)
- `--imputation`: mean, median, zero (default: median)
- `--scaling`: min-max, standard, robust (default: min-max)
- `--encoding`: one-hot, ordinal (default: one-hot)
- `--help` (show help message)

**Example:**

```bash
./build/autism_screening \
  --input data/Autism-Adult-Data.csv \
  --output data/processed.bin \
  --bins 5 \
  --imputation mean \
  --scaling robust \
  --encoding ordinal
```

## Data Processing
- CSV loading with validation.
- Missing values represented by std::optional.
- Automatic feature type detection.
- NumericColumn and CategoricalColumn stored using std::variant.
- Statistics: min, max, mean, variance, Q05, Q25, Q50, Q75, Q95.
- Interactive feature selection and text histograms.
- Missing value imputation: mean/median/zero.
- Scaling: MinMax/Standard/Robust.
- Encoding: OneHot/Ordinal.
- Class/ASD stored separately as target labels.
- Output is a row-major matrix of doubles.

## Binary Format
- Saves matrix dimensions, column names, double values and target labels.
- File signature ASML, version 1, row-major layout.
- Reloads the file and verifies equality.
- One-hot encoding can make BIN larger than CSV.

## Testing
- **CTest**: `ctest --test-dir build --output-on-failure`
- **Standalone tests**: `./build/autism_tests`
- 10 test groups, including CSV errors, statistics, encoding, all 18 preprocessing combinations and binary round-trip.

## Sanitizers
- ASan and UBSan supported through ENABLE_SANITIZERS.
- Commands:
  - `cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON`
  - `cmake --build build-sanitize`
  - `ctest --test-dir build-sanitize --output-on-failure`

## Limitations
- Age contains an outlier of 383.
- Missing values in CSV use '?'.
- Binary format is not guaranteed portable across architectures.
- MinMax, Standard and Robust scaling return 0 when the corresponding denominator is zero.


## Lab 2 — Decision Stump

This laboratory implements a depth-one decision tree for regression and classification using the common `IModel` interface. The model selects the best feature and threshold using variance-based splitting for regression and Gini impurity for classification.

The naive threshold search has a time complexity of O(d × n²). The optimized implementation reduces it to O(d × n log n) with O(n) auxiliary memory by sorting object indices once per feature and updating the splitting criterion in O(1) for each candidate threshold.

The implementation was verified against a naive reference algorithm and passed CTest, including Debug tests with AddressSanitizer and UndefinedBehaviorSanitizer.

On the Autism Screening Adult dataset, the model selected the `result` feature with a threshold of 0.65 after MinMax scaling. Both regression and classification produced the same split.

## Lab 3 — CART Decision Tree (Star Track)

This laboratory implements CART decision trees for regression and
classification. It extends the decision stump from Lab 2 by
recursively applying the shared `SplitFinder` to subsets of
training objects.

### Architecture

- `MatrixSlice<T>` provides non-owning row and column views with
  a pointer, size, stride, and slice direction.
- `NumericMatrix` supports RowMajor and ColumnMajor storage.
- `IRegressor` and `IClassifier` extend the common `IModel`
  interface; classifiers also implement `predict_proba()` and
  `classes()`.
- `SplitFinder` searches for the best numerical threshold and
  supports subsets of objects and candidate features.
- `CartTree` implements the shared recursive training and
  prediction logic.
- `CartRegressor` and `CartClassifier` provide task-specific
  model interfaces.

Each tree owns its root through `std::unique_ptr`. Internal nodes
own two children; leaf nodes store a prediction and, for
classification, class probabilities. Recursion passes object
indices without copying the original feature matrix.

Matrix slices do not own their elements and must not outlive the
matrix. A contiguous slice can be converted to `std::span`.
Non-contiguous slices remain accessible through their stride,
but conversion to `std::span` throws an exception.

### Training and Prediction

CART supports three main hyperparameters:

- `max_depth`: maximum tree depth, with the root at depth 0.
- `min_samples_split`: minimum objects required to attempt a split.
- `min_samples_leaf`: minimum objects in each resulting leaf.

Training stops when a stopping condition is reached, all targets
are identical, or no valid split exists. The split search uses
the regression variance criterion or the classification Gini
criterion. Equal split scores are resolved deterministically by
feature index and threshold.

Regression leaves return the mean target value. Classification
leaves predict the most frequent class and store class proportions
in the order returned by `classes()`. Class ties are resolved in
favour of the smallest numerical label.

Successful repeated calls to `fit()` replace the previous trained
state. Invalid training data is rejected before the new tree
replaces the old one.

### Complexity

At a node containing `s` objects and `d` candidate features,
split search costs `O(d * s * log(s))`, including sorting.
Total training complexity is the sum over all internal nodes.

Prediction requires `O(depth)` comparisons per object.
The implementation does not construct a new feature matrix for
each recursive node.

### Demonstration

Build and run:

```bash
cmake -S . -B build
cmake --build build
./build/cart_demo
```

The Autism Screening Adult classification demonstration uses
563 training objects, 141 test objects, and 113 processed features.

With `max_depth=1`, the resulting tree has three nodes and
two leaves. It splits on `result <= 0.65` after scaling.

With `max_depth=4`, the actual depth remains 1 because the
same split already separates the training classes.

The independent synthetic regression example grows to depth 2,
with seven nodes and four leaves. It demonstrates recursive
partitioning beyond a decision stump.

### Testing

Run all registered tests:

```bash
ctest --test-dir build --output-on-failure
```

The test suite covers the preprocessing pipeline, the decision
stump, matrix slices, and CART regression and classification.

Debug sanitizers can be enabled with:

```bash
cmake -S . -B build-sanitize \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_SANITIZERS=ON
cmake --build build-sanitize
ctest --test-dir build-sanitize --output-on-failure
```

### Limitations

- Only binary splits on numerical feature columns are supported.
- Pruning and categorical subset splits are not implemented.
- Leaf probabilities are class frequencies, not calibrated
  confidence estimates.
- Floating-point split comparisons have finite precision.
- Class labels are encoded as numbers for the model interface;
  the original `Class/ASD` values remain available in the
  dataset's target labels.
- RowMajor and ColumnMajor offer equivalent logical access,
  but have different memory access patterns.