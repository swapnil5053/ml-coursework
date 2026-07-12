# Machine Learning Lab

Coursework from my Machine Learning lab, written up as clean, executed Jupyter
notebooks - each one implements a method from the ground up before reaching for
the library version, so the mechanics are visible. Alongside the labs is a larger
time-series forecasting project.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E)

## Labs

| Folder | Topic | The idea |
|--------|-------|----------|
| [`lab01-decision-trees`](lab01-decision-trees) | Decision trees | Entropy, information gain and ID3 from scratch, checked against scikit-learn |
| [`lab02-model-selection`](lab02-model-selection) | Model selection | Manual grid search vs. `GridSearchCV`, stratified k-fold, DT / k-NN / LogReg |
| [`lab03-svm-kernels`](lab03-svm-kernels) | SVM kernels | Linear / poly / RBF decision boundaries on non-linear and real data |
| [`lab04-naive-bayes`](lab04-naive-bayes) | Naive Bayes | Multinomial NB from scratch + TF-IDF + a voting ensemble for text |

Every notebook is committed **with its outputs** - the plots and numbers you see
are from an actual run, and each one loads its data from scikit-learn or generates
it, so they reproduce without any download.

## Mini-project

[`wikitraffic-forecasting`](wikitraffic-forecasting) - forecasting daily
Wikipedia page views and comparing a KNN baseline, an LSTM, and a seq2seq causal
CNN, scored with SMAPE. The included notebook walks the feature pipeline and the
KNN baseline end to end; the two deep models train through `src/main.py`.

## Running a lab

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook lab02-model-selection/grid_search_model_selection.ipynb
```

Each folder has a short README with specifics.

## Tech

Python, NumPy, pandas, scikit-learn, matplotlib/seaborn, Jupyter. The forecasting
project also uses TensorFlow for the deep models.

---
Swapnil Shantha Kumar - PES University
