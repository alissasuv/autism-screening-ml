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
