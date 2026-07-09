# Model Selection: Grid Search and Cross-Validation

Hyperparameter tuning done the honest way - stratified 5-fold CV, ROC AUC, and a
pipeline that refits the scaler inside every fold. I wrote the grid search by hand
first, then reproduced it with `GridSearchCV` and confirmed the two agree, across
a Decision Tree, k-NN, and Logistic Regression.

## What's here

- a from-scratch `manual_grid_search` over stratified folds
- the equivalent `GridSearchCV` setup on the same pipeline and metric
- a side-by-side table and chart showing the two methods match
- Breast Cancer Wisconsin dataset (loads from scikit-learn, no download)

## Run

```bash
jupyter notebook grid_search_model_selection.ipynb
```

Needs `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
