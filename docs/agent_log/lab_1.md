# Lab 1 — AI Assistance Log

## Tools Used

OpenCode with Qwen3-4B-Instruct-2507, running locally via LM Studio.

## Task

AI was used to generate and revise the initial project documentation (`README.md`). The dataset and feature type detection rule were selected independently.

## Prompt 1 — Initial README Generation

```text
Fill in the existing README.md for my educational C++20 machine learning project.

Dataset: Autism Screening Adult (UCI), 704 records, 20 features and one target column.

Columns: A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score, A10_Score, age, gender, ethnicity, jundice, austim, contry_of_res, used_app_before, result, age_desc, relation, Class/ASD.

Missing values are represented by ?. Missing counts: age 2, ethnicity 95, relation 95.

The first 10 columns are binary AQ-10 questionnaire scores. The target Class/ASD contains screening classifications, not confirmed medical diagnoses.

My feature type detection rule: non-numeric features and numeric features with at most 2 unique values are categorical. Other numeric features are numerical. Ignore missing values.

Dataset source: https://archive.ics.uci.edu/dataset/426/autism+screening+adult

Write a concise professional English README with a project overview, dataset description, feature table, missing values and type detection rule. The C++20 implementation has not started yet.

Do not inspect the dataset, browse the web or run terminal commands. Only edit README.md.
```

**Result:** Qwen generated `README.md` with the dataset overview, feature table, missing-value counts, and feature type detection rule.

**Manual review:** Three inaccuracies were identified: `result` was classified as categorical, `austim` was described incorrectly, and the overview implied that a machine learning model had already been implemented.

## Prompt 2 — README Corrections

```text
Fix three inaccuracies in README.md. The result feature is Numerical and represents the total AQ-10 screening score. The austim feature refers to family history, not the participant's own diagnosis. The project overview must not claim that a machine learning model has already been implemented.

Keep everything else unchanged. Edit only README.md.
```

**Result:** Qwen corrected the feature types and descriptions and changed the project overview to describe future implementation.

## Prompt 3 — README Documentation Update

**Tool:** OpenCode with Qwen3-4B-Instruct-2507, running locally via LM Studio.

**Prompt:**

```text
Update README.md for my C++20 ML Lab 1 project.

Keep the existing dataset description unchanged.
Write everything in English.
Do not read other source files. Do not modify any files except README.md.

Add these sections:

1. Build and Run
- Requirements: C++20, CMake, Clang/GCC.
- Build: cmake -S . -B build && cmake --build build
- Run: ./build/autism_screening
- Show how to use --help.

2. CLI Options
Document:
--input (default data/Autism-Adult-Data.csv)
--output (default data/processed.bin)
--bins (default 10)
--imputation: mean, median, zero (default median)
--scaling: min-max, standard, robust (default min-max)
--encoding: one-hot, ordinal (default one-hot)
--help

Include one example command with custom options.

3. Data Processing
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

4. Binary Format
- Saves matrix dimensions, column names, double values and target labels.
- File signature ASML, version 1, row-major layout.
- Reloads the file and verifies equality.
- Explain that one-hot encoding can make BIN larger than CSV.

5. Testing
- CTest: ctest --test-dir build --output-on-failure
- Standalone tests: ./build/autism_tests
- 10 test groups, including CSV errors, statistics, encoding,
  all 18 preprocessing combinations and binary round-trip.

6. Sanitizers
- ASan and UBSan supported through ENABLE_SANITIZERS.
- Commands:
  cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
  cmake --build build-sanitize
  ctest --test-dir build-sanitize --output-on-failure

7. Limitations
- Age contains an outlier of 383.
- Missing values in CSV use '?'.
- Binary format is not guaranteed portable across architectures.
- MinMax, Standard and Robust scaling return 0 when
  the corresponding denominator is zero.

Use clean Markdown headings and short explanations.
Do not invent any other features.
Do not create a commit.
```

**Result:** Qwen expanded `README.md` with build instructions, CLI options, data processing stages, binary serialization, automated testing, sanitizers, and known limitations.

**Manual review:** Several issues were identified and corrected:

1. The project overview was outdated and described future implementation instead of the existing data processing pipeline.
2. The CLI example was reformatted and replaced with a command using previously tested parameters.
3. The binary output description was clarified: the file is created after the user confirms preprocessing by entering `y`.

**Conclusion:** The generated README required several manual corrections to accurately reflect the implemented functionality. The AI was used to update documentation only; no C++ source files were modified during this task.

## Verification and Conclusions

The resulting README was reviewed manually. The feature names, missing-value counts, and feature type detection rule were checked for consistency with the dataset information.

AI assistance was limited to documentation. No C++ algorithms or program implementation were generated by Qwen.