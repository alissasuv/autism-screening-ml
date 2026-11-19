# Lab 5: Hyperparameter Tuning and Reproducible Experiments

## 1. Objective

The objective of this laboratory work is to implement a reproducible
machine-learning experiment with separate data preparation, model tuning,
and final evaluation stages.

The advanced track compares two families of random forests:

- Deep forests with unrestricted tree depth.
- Shallow forests with tree depth limited to 2 or 3.

The experiment includes fold-safe preprocessing, grid search,
cross-validation, train/test evaluation, and permutation feature
importance calculated over original input features.

A secondary regression experiment verifies that the same infrastructure
supports another prediction task.

## 2. Dataset and Prediction Task

### 2.1 Main dataset

The main experiment uses the UCI Autism Screening Adult dataset.

Source:
https://archive.ics.uci.edu/dataset/426/autism+screening+adult

The dataset contains 704 observations. The target column is `Class/ASD`,
a binary screening classification label with values `YES` and `NO`.

This label should not be interpreted as a medical diagnosis.

The dataset includes questionnaire responses `A1_Score` through
`A10_Score`, demographic attributes, screening-related information,
and the numerical `result` attribute.

The `result` attribute is particularly important when interpreting
the experiment, because it is a questionnaire-derived summary that
is closely related to the screening label.

### 2.2 Train/test split

The original dataset was split into:

| Partition | Observations |
|---|---:|
| Train | 563 |
| Test | 141 |
| Total | 704 |

The classification split is stratified, uses a test fraction of 0.2,
and is reproducible using `random_state=42`.

The raw partitions were saved as `data/train.bin` and `data/test.bin`.
The dataset statistics were saved in `results/dataset_stats.json`.

Model selection was performed exclusively on the training partition.
The test partition was reserved for final evaluation.

## 3. Preprocessing Protocol

Preprocessing is configured in `configs/preprocessing.yaml`.

The experiment uses:

| Data type | Operation | Configuration |
|---|---|---|
| Numeric | Missing-value imputation | Median |
| Numeric | Scaling | Standard |
| Categorical | Missing values | Dedicated missing category |
| Categorical | Encoding | One-hot |

Feature types are detected using training data only.

During cross-validation, preprocessing is fitted separately on the
training portion of each fold. The validation portion is transformed
using the learned statistics and category dictionaries.

During final evaluation, preprocessing is fitted on the complete
training partition and then applied unchanged to train and test.

The category vocabulary and output-column order remain fixed after
fitting. Previously unseen categories are represented by zero values
in the corresponding one-hot group.

This protocol prevents preprocessing statistics or category dictionaries
from being learned from validation or test data.

## 4. Model Families

Both model families reuse the Random Forest implementation developed
in Lab 4. No external machine-learning library is used to train models.

The deep forest allows unrestricted tree depth. The shallow forest
restricts tree depth to 2 or 3.

Both use bootstrap sampling and a reproducible random seed.

The number of trees and minimum leaf size are included in the search
space. The shallow forest additionally searches over maximum depth.

### 4.1 Search configurations

Deep forest:

- `max_depth: null`
- `n_estimators: [10, 25]`
- `min_samples_leaf: [1, 3]`
- 4 candidates

Shallow forest:

- `max_depth: [2, 3]`
- `n_estimators: [10, 25]`
- `min_samples_leaf: [1, 3]`
- 8 candidates

The remaining settings and the complete candidate definitions are
recorded in:

- `configs/search_deep_forest.yaml`
- `configs/search_shallow_forest.yaml`

The exact selected parameters are stored in:

- `results/best_deep_forest.yaml`
- `results/best_shallow_forest.yaml`

These saved files are the authoritative records of the selected
hyperparameters.

## 5. Model Selection

Grid search was used with three stratified cross-validation folds,
`random_state=17`, and macro-F1 as the selection metric.

All candidates within both families used the same fold assignment.
A separate model and preprocessing state were fitted for every fold.

The experiment evaluated 12 candidates in total:

| Family | Candidates | CV folds | Model fits |
|---|---:|---:|---:|
| Deep | 4 | 3 | 12 |
| Shallow | 8 | 3 | 24 |
| Total | 12 | — | 36 |

### 5.1 Cross-validation results

| Family | Best candidate | Mean macro-F1 | CV standard deviation |
|---|---:|---:|---:|
| Deep | 3 | 0.997715 | 0.003232 |
| Shallow | 7 | 0.983711 | 0.011886 |

The unrestricted-depth forest achieved a higher mean macro-F1 and a
smaller standard deviation across the three validation folds.

The full search histories, including unsuccessful configurations,
are saved in the corresponding `*_search.csv` files.

The best configuration was selected independently within each model
family rather than choosing only one overall winner.

## 6. Final Classification Evaluation

After hyperparameter selection, each model was trained again on the
complete training partition.

No hyperparameters were changed based on the test results.

### 6.1 Train and test metrics

| Model | Split | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
|---|---|---:|---:|---:|---:|
| Deep | Train | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Deep | Test | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Shallow | Train | 0.973357 | 0.982436 | 0.950331 | 0.964928 |
| Shallow | Test | 0.978723 | 0.985849 | 0.960526 | 0.972275 |

The deep forest correctly classified all 141 test observations.
The shallow forest correctly classified 138 of 141 observations.

The deep forest therefore performed better on the held-out test
partition under all four reported classification metrics.

### 6.2 Train/test comparison

The deep forest achieved perfect accuracy on both partitions.
The shallow forest's test macro-F1 was slightly higher than its
training macro-F1.

The latter observation does not, by itself, indicate a mistake.
Different data partitions may have different levels of difficulty.

The deep forest's perfect result must not be treated as proof of
general diagnostic accuracy. The screening target is closely related
to questionnaire-derived information already included among the
predictors, particularly `result`.

### 6.3 Model complexity

The deep and shallow forests differ in their allowed tree complexity.
On this dataset, restricting depth reduced the observed classification
performance.

This is consistent with a possible underfitting effect in the shallow
family. However, the experiment does not isolate tree depth from all
other chosen hyperparameters, so it does not prove that depth alone
caused the entire performance gap.

The unrestricted-depth forest did not show a measurable train/test
performance gap on the selected test partition. This does not rule
out overfitting on other datasets.

## 7. Permutation Feature Importance

Permutation feature importance was computed on the held-out test set
using the previously fitted model and preprocessing state.

Each original feature column was shuffled independently five times,
using `random_state=123`.

Missing values were shuffled together with the other raw feature
values. Preprocessing was applied only after permutation.

The target was never shuffled.

Importance is defined as the deterioration in the scoring metric after
permuting a feature. For macro-F1, a positive importance value indicates
that performance decreased.

### 7.1 Main feature-importance results

| Feature | Deep importance | Shallow importance |
|---|---:|---:|
| result | 0.146446 | 0.110211 |
| A9_Score | 0.007266 | 0.039560 |
| A6_Score | 0.003573 | 0.020940 |
| A10_Score | 0.001816 | 0.020503 |
| A5_Score | 0.012778 | 0.005746 |
| ethnicity | 0.001816 | 0.011528 |

The `result` feature has the highest mean permutation importance
for both forests.

For the deep forest, its average permutation decreases macro-F1
by approximately 0.1464. For the shallow forest, the corresponding
decrease is approximately 0.1102.

The two forests distribute importance differently across individual
questionnaire features. The shallow forest shows relatively higher
importance for `A9_Score`, `A6_Score`, and `A10_Score`.

### 7.2 Limitations of permutation importance

Permutation importance measures the model's dependence on a feature
under the current dataset and fitted model. It is not a causal measure
of the feature's effect.

Correlated features may provide overlapping information. Consequently,
permuting one feature can have little effect even when that feature
contains useful predictive information.

A zero importance value does not prove that a feature has no predictive
value.

The reported estimates also depend on the test sample and the
particular random permutations. Five repetitions provide a limited
estimate of their variability.

The importance of `result` must be interpreted in light of its close
relationship with the target definition.

Complete importance tables are available in:

- `results/deep_forest_importance.csv`
- `results/shallow_forest_importance.csv`

## 8. Secondary Regression Smoke Test

### 8.1 Synthetic data

A separate synthetic dataset of 24 observations was created with two
input attributes:

- `x`: a numerical predictor.
- `category`: a categorical predictor.

The response `target` is numerical and generated deterministically
from both input attributes.

The dataset is stored in `data/synthetic_regression.csv`.

The smoke test uses 18 training observations and 6 test observations.
Every fourth observation is assigned to the test partition.

Two shuffled CV folds are used during model selection.
The selection metric is mean squared error (MSE).

### 8.2 Regression search budget

| Family | Candidates | Folds | Model fits |
|---|---:|---:|---:|
| Deep | 2 | 2 | 4 |
| Shallow | 4 | 2 | 8 |

### 8.3 Regression results

| Metric | Deep Forest | Shallow Forest |
|---|---:|---:|
| Best CV MSE | 33.9904 | 34.2412 |
| Train MSE | 3.94792 | 3.76612 |
| Test MSE | 10.9063 | 11.6385 |
| Train MAE | 1.59722 | 1.67191 |
| Test MAE | 2.65278 | 2.89884 |
| Train R2 | 0.990755 | 0.991181 |
| Test R2 | 0.975272 | 0.973611 |

Both models achieved high R2 values on this synthetic task.

The deep forest performed slightly better on test MSE, MAE, and R2.
The shallow forest achieved a slightly lower training MSE.

The test errors were higher than the training errors for both models.
Because the test partition contains only six observations, these
differences should not be overinterpreted.

The smoke test also successfully exercised:

- YAML configuration loading.
- Grid search and two-fold cross-validation.
- Final training and prediction.
- Regression metrics.
- Permutation importance.
- Saving search, metric, prediction, and importance files.

This regression check uses an in-memory deterministic train/test split,
rather than the classification-oriented `prepare_data` CLI.

## 9. Reproducibility and Validation

The project is built with C++20 and CMake.

The main data split, CV partitioning, random forests, and permutation
importance all use explicitly configured random seeds.

All 17 registered CTest tests passed during the final run.

The test suite includes checks for:

- Raw data persistence and splitting.
- Fold-safe preprocessing.
- Classification and regression metrics.
- Random Forest model creation.
- Cross-validation and reproducibility.
- Grid search and configuration persistence.
- Final evaluation and prediction export.
- Permutation importance.
- End-to-end regression smoke testing.

The saved CSV and YAML artifacts support inspection and reproduction
of the experiments.

## 10. Limitations

The experiment has several limitations.

First, the main dataset is comparatively small.

Second, `result` is strongly related to the screening target. The
near-perfect classification results may therefore primarily reflect
the structure of the questionnaire-derived target rather than
generalization to an independent clinical outcome.

Third, the grid search covers a deliberately limited number of
hyperparameter combinations.

Fourth, the experiment uses a single held-out test partition.
Performance variability under other train/test splits has not been
measured.

Fifth, permutation importance can underestimate importance when
features are correlated.

Finally, the regression experiment is only a functional smoke test
with a small synthetic dataset, not a performance benchmark.

## 11. Conclusion

The experiment successfully implemented a reproducible pipeline for
Random Forest hyperparameter selection and evaluation.

The deep forest outperformed the shallow forest on the main
classification task and slightly outperformed it on the synthetic
regression test.

The main classification test macro-F1 scores were 1.000000 and
0.972275, respectively.

Permutation importance identified `result` as the most influential
original feature in both classification models. This finding is
important but also limits the interpretation of the very high
classification scores.

The secondary regression smoke test confirmed that the shared
model-tuning and evaluation infrastructure supports numerical targets.

These findings describe the observed experimental results and should
not be generalized to medical diagnostic performance.
