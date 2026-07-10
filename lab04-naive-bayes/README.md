# Naive Bayes for Text Classification

Classifying medical-abstract sentences by role (BACKGROUND / METHODS / RESULTS /
CONCLUSIONS) - the PubMed 20k RCT task. Because that corpus needs a download, the
notebook generates a self-contained one where each role is a multinomial over a
shared medical vocabulary, with overlap built in between related roles so the
task stays honest.

## What's here

- a Multinomial Naive Bayes classifier written from scratch (Laplace smoothing, log-space scoring)
- the scikit-learn TF-IDF version tuned with `GridSearchCV`
- a soft-voting ensemble (NB + Logistic Regression + Random Forest)
- a confusion matrix that lands exactly on the overlapping role pairs

Results sit in the high-80s to low-90s, with the residual confusion between
BACKGROUND/CONCLUSIONS and METHODS/RESULTS - by construction, and the way real
abstracts behave too.

## Run

```bash
jupyter notebook naive_bayes_text_classification.ipynb
```

Needs `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
