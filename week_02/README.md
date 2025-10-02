# Week 02 — Regression & Data Preprocessing

🎯 **Goal**  
Learn the basics of scikit-learn and classification via logistic regression.  
Build a reproducible preprocessing pipeline (imputation, scaling, encoding) and apply it to the Titanic dataset.  
Produce clear evaluation with proper metrics.

---

## Daily Plan

- [ ] **Day 8 — scikit-learn fundamentals**
    * Install scikit-learn and read the Quickstart.
    * Understand the Estimator API (fit, transform, predict), Transformers, Pipelines, ColumnTransformer.
    * Beginner: identify Titanic features by type (numeric vs categorical).
    * Advanced: explore `make_classification` for synthetic data.
    * References:
        - https://scikit-learn.org/stable/getting_started.html
        - https://scikit-learn.org/stable/supervised_learning.html
        - https://scikit-learn.org/stable/modules/compose.html#pipeline

- [ ] **Day 9 — Titanic feature engineering (data audit)**
    * Verify dataset (columns, dtypes, target, missing values).
    * Handle missing data: Age (median), Fare (median), Embarked (most frequent).
    * Beginner: add FamilySize, IsAlone, AgeGroup.
    * Advanced: create interaction features (e.g., Sex × Pclass) or Fare bins.
    * References:
        - https://pandas.pydata.org/docs/user_guide/missing_data.html
        - https://pandas.pydata.org/docs/user_guide/categorical.html

- [ ] **Day 10 — Preprocessing pipeline**
    * Build preprocessing with scikit-learn:
        - Numeric: SimpleImputer + StandardScaler.
        - Categorical: SimpleImputer + OneHotEncoder.
        - Combine with ColumnTransformer.
    * Beginner: split features into numeric and categorical lists.
    * Advanced: test StandardScaler vs MinMaxScaler.
    * References:
        - https://scikit-learn.org/stable/modules/preprocessing.html
        - https://scikit-learn.org/stable/modules/impute.html
        - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

- [ ] **Day 11 — Baselines and logistic regression**
    * Train DummyClassifier for baseline.
    * Train LogisticRegression with preprocessing pipeline.
    * Compare baseline vs real model.
    * Beginner: run default LogisticRegression.
    * Advanced: use `class_weight="balanced"` and compare.
    * References:
        - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        - https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html

- [ ] **Day 12 — Evaluation metrics**
    * Compute Accuracy, Precision, Recall, F1-score.
    * Visualize confusion matrix.
    * Beginner: text confusion matrix.
    * Advanced: heatmap + ROC/PR curves.
    * References:
        - https://scikit-learn.org/stable/modules/model_evaluation.html
        - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

- [ ] **Day 13 — Interpretation and error analysis**
    * Inspect logistic regression coefficients and link to feature names.
    * Beginner: list top positive/negative contributors.
    * Advanced: threshold tuning, precision-recall trade-off.
    * References:
        - https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
        - https://scikit-learn.org/stable/modules/calibration.html

- [ ] **Day 14 — Wrap-Up & Documentation**
    * Summarize preprocessing design, baseline vs model metrics, insights.
    * Save plots to `week_02/plots/`.
    * Update README with reflections.
    * Commit notebook, plots, and cleaned dataset.
    * References:
        - https://scikit-learn.org/stable/modules/model_persistence.html

---

## References & Learning Resources
- Hands-On Machine Learning (chapters on training models and classification)
- Python Data Science Handbook — scikit-learn chapter: https://jakevdp.github.io/PythonDataScienceHandbook/
- scikit-learn Quickstart: https://scikit-learn.org/stable/getting_started.html
- scikit-learn Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
- Pipelines and ColumnTransformer: https://scikit-learn.org/stable/modules/compose.html
- LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html

---

## Deliverables for Week 2
- Notebook(s) with:
    * Preprocessing pipeline for Titanic dataset.
    * Baseline (DummyClassifier) and LogisticRegression experiments.
    * Evaluation with accuracy, precision, recall, F1.
    * Confusion matrix (text/plot), optional ROC/PR curves.
- Cleaned dataset (CSV).
- Plots in `week_02/plots/`.
- Updated README with progress notes and references.
- GitHub commits showing daily work.
