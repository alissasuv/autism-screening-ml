# Autism Screening Adult Machine Learning Project


## Project Overview

This project is an educational C++20 machine learning implementation
based on the UCI Autism Screening Adult dataset.

The repository is developed incrementally through laboratory assignments.
It includes a complete data preprocessing pipeline, decision stumps,
CART decision trees, and Random Forest models for regression and
classification.

All machine learning algorithms are implemented in C++ without
third-party machine learning libraries. The project uses CMake,
automated tests through CTest, and optional sanitizers.

The Random Forest implementation supports bootstrap sampling,
random feature selection at each tree node, deterministic random
seeds, probability averaging for classification, and parallel
training and prediction using C++20 std::jthread.

The project is intended for educational purposes. The dataset's
Class/ASD target represents a screening classification, not a
confirmed clinical diagnosis.


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


## Lab 4 — Random Forest (Star Track)

### Overview

Lab 4 implements Random Forest models for regression and
classification using the CART trees developed in Lab 3.

The implementation follows the core algorithmic principles of
scikit-learn 1.8.0 RandomForestRegressor and
RandomForestClassifier. It does not reproduce the complete
scikit-learn API or its exact random number sequences.

The forest combines multiple CART estimators trained with
independent random seeds and, optionally, bootstrap samples.

### Architecture

The main components are:

- `RandomForestParameters`: forest configuration.
- `RandomForestRegressor`: regression ensemble implementing IRegressor.
- `RandomForestClassifier`: classification ensemble implementing IClassifier.
- `CartRegressor` and `CartClassifier`: reusable base estimators.
- `SplitFinder`: weighted numerical split search.
- `forest_parallel::for_each_index`: bounded parallel task execution.

The forest owns its trees through a vector of std::unique_ptr
instances. Each tree owns its nodes recursively through
std::unique_ptr.

The forest uses composition rather than inheritance from CART.
Both standalone CART and forest estimators share the same tree
building and split-search implementations.

No external machine learning algorithms or metrics libraries
are used.

### Hyperparameters

| Parameter | Description |
|---|---|
| n_estimators | Number of trees; default: 100 |
| bootstrap | Enable sampling with replacement; default: true |
| max_samples | Optional number of bootstrap draws |
| max_features | All, Sqrt, or an explicit feature count |
| max_depth | Optional tree depth limit |
| min_samples_split | Minimum number of distinct source objects required to attempt a split |
| min_samples_leaf | Minimum number of distinct source objects per child |
| random_state | Global seed for reproducible forest construction |
| n_jobs | Maximum number of worker threads |

By default, classification uses Sqrt feature sampling, whereas
regression uses All features.

### Bootstrap Sampling

For each tree, bootstrap sampling draws training indices with
replacement. By default, the number of draws equals the
number of training objects.

Repeated indices are represented by integer multiplicities
rather than physically duplicated matrix rows.

For an original object i, its multiplicity is count[i].

The implementation distinguishes:

- The number of distinct source indices in a node.
- The total weight, equal to the sum of multiplicities.

The min_samples_split and min_samples_leaf constraints use
the number of distinct source indices.

Regression means, class frequencies, and split criteria use
the bootstrap multiplicities.

A class may be absent from an individual bootstrap sample.
This is a valid outcome and is not corrected by resampling.

When bootstrap is disabled, every training object has unit
multiplicity.

### Random Feature Selection

At each CART node, a fresh ordering of candidate features
is generated according to the configured feature-search policy.

The supported modes are:

- All: consider every feature.
- Sqrt: consider approximately sqrt(d) features.
- Count: consider an explicitly configured number of features.

If the initial candidate features are constant, the search
continues until a nonconstant feature is considered or all
features have been exhausted.

For each considered feature, SplitFinder evaluates valid
thresholds and selects the best split.

The threshold itself is not selected randomly.

### Weighted Split Criteria

For regression, the split search maximizes:

    S_left^2 / W_left + S_right^2 / W_right

where S is the weighted target sum and W is the total
bootstrap multiplicity.

For classification, the split search maximizes:

    A_left / W_left + A_right / W_right

where A is the sum of squared weighted class counts.

Both criteria use the same sorted-threshold search mechanism
as the earlier CART implementation.

The split search has O(d * n * log(n)) complexity at a node
when all d features are considered.

### Regression Predictions

Each tree produces a numerical prediction.

The forest prediction is the arithmetic mean of the
individual tree predictions:

    prediction(x) = sum(tree_b(x)) / B

where B is the number of trees.

### Classification Predictions

Each classification tree returns class probabilities,
calculated from weighted class frequencies in its leaf.

The forest averages probabilities across trees:

    P(class=c | x) = sum(P_b(class=c | x)) / B

The final predicted class is the class with the highest
average probability.

The algorithm does not use majority voting over tree labels.

The global class order is determined from the complete
training target and returned by classes().

If a bootstrap sample does not contain a particular class,
that tree contributes zero probability for the missing class.

Probability columns are aligned with the forest's global
class order before averaging.

In the event of a tie, the smallest numerical class label
is selected.

### Parallel Execution

Parallel training and prediction use C++20 std::jthread.

The n_jobs parameter limits the number of worker threads.
The actual number of workers does not exceed the number
of trees.

For n_jobs=1, tasks execute sequentially.

Each worker processes a deterministic subset of tree indices.
The input feature matrix and target vector are shared
read-only.

Each tree is constructed and trained independently.
Workers write into separate, preallocated result slots.

No shared mutable random number generator is used.

Exceptions raised by worker tasks are captured as
std::exception_ptr values. After all worker threads have
finished, exceptions are rethrown in the calling thread.

Predictions are accumulated in tree-index order rather than
thread-completion order.

This makes results reproducible within the same execution
environment when changing n_jobs.

### Random Seeds and Reproducibility

The forest initializes a master std::mt19937 generator from
random_state.

Before parallel execution, it generates distinct seeds
for bootstrap sampling and feature selection.

Each tree uses its assigned seeds independently.

Calling fit() again with unchanged inputs and parameters
reconstructs the same random sequence and trained ensemble
within the same environment.

Different numbers of worker threads do not change the
assignment of seeds to trees.

Exact agreement with scikit-learn's random forests is not
expected because the implementations use different random
number generation and internal tie-breaking details.

### Training Lifecycle

The forest constructor validates parameters that do not
depend on the training data.

fit() validates the input and constructs a complete new
ensemble.

Successful training replaces the previous forest rather
than appending additional trees.

If a refit fails, the previous trained ensemble remains
available.

Prediction does not modify the trained forest.

Calling predict() before successful training raises an
exception.

### Computational Complexity

Let:

- B be the number of trees.
- n be the number of training objects.
- d be the number of features.
- q be the number of candidate features considered per node.
- h be the tree depth.

For a node containing s distinct source objects, the
candidate split search requires approximately:

    O(q * s * log(s))

The total training cost is the sum of this work across
all internal nodes of all B trees.

For balanced trees and a fixed feature budget, a rough
training estimate is:

    O(B * q * n * log(n)^2)

This is an estimate rather than a strict bound for
arbitrary tree shapes.

Prediction for m objects requires approximately:

    O(B * m * h)

The implementation stores the original feature matrix
once and passes source indices and multiplicities to
the CART training logic.

Parallel execution may reduce wall-clock time but does
not reduce the total algorithmic work. Additional memory
is required for independent prediction buffers.

### Demonstration

The Random Forest demonstration compares standalone CART
and Random Forest classification using the same train/test
split of the Autism Screening Adult dataset.

It reports:

- Training and test sample counts.
- Number of processed features.
- Classification accuracy.
- Several predictions and their probabilities.
- Tree node counts and depths.
- Training and prediction times.
- Sequential and parallel performance.

A separate synthetic example demonstrates regression.

Build and run:

```bash
cmake -S . -B build
cmake --build build -j 4

./build/random_forest_demo data/Autism-Adult-Data.csv
```

For meaningful timing measurements, use a Release build
without sanitizers:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_SANITIZERS=OFF

cmake --build build-release -j 4

./build-release/random_forest_demo \
  data/Autism-Adult-Data.csv
```

The demonstration measures preprocessing separately from
model training and prediction.

Training and prediction times are measured independently.
Three repetitions are used, and the median is reported.

Parallel speedup is defined as:

    speedup = sequential_time / parallel_time

Parallel execution may be slower on small datasets because
thread creation and result aggregation introduce overhead.


### Testing

The project includes dedicated test executables for:

- Data loading and preprocessing.
- Decision Stump.
- Matrix slices.
- CART.
- Weighted SplitFinder.
- Weighted CART.
- Random Forest regression.
- Random Forest classification.
- Parallel Random Forest execution.

The tests cover reproducibility, input validation,
weighted splits, missing bootstrap classes, single-tree
equivalence, both matrix layouts, repeated fitting,
prediction before training, and exception propagation.

Run the test suite:

```bash
cmake -S . -B build
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

Debug validation with AddressSanitizer and
UndefinedBehaviorSanitizer:

```bash
cmake -S . -B build-sanitize \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_SANITIZERS=ON

cmake --build build-sanitize -j 4

ctest --test-dir build-sanitize \
  --output-on-failure
```

ThreadSanitizer may be run separately when supported
by the compiler and platform.

### Limitations

- Only finite, dense numerical feature matrices are supported.
- Missing values must be handled before model training.
- Multi-output targets and sparse matrices are not supported.
- Native NaN handling is not implemented.
- Sample weights other than internal bootstrap multiplicities
  are not supported.
- OOB scoring and impurity-based feature importance are not
  implemented.
- Pruning, warm start, and monotonic constraints are omitted.
- Exact equivalence with scikit-learn's random sequences and
  internal tree structures is not guaranteed.
- Parallel speedup depends on dataset size and hardware.
- Screening classification results must not be interpreted
  as clinical diagnostic performance.

The laboratory focuses on implementing the core Random
Forest algorithm and demonstrating its correctness,
reproducibility, and execution behavior.
