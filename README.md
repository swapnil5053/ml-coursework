# Machine Learning Coursework

A collection of my Machine Learning laboratory assignments and practical courseworks from my University. Each lab is implemented as a Jupyter notebook and focuses on understanding the underlying algorithms before using the corresponding scikit-learn implementation.

Along with the lab exercises, this repository also includes a larger time-series forecasting project completed as part of the course.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E)

---

## Repository Structure

| Folder | Topic | Description |
|--------|-------|-------------|
| [`lab01-decision-trees`](lab01-decision-trees) | Decision Trees | Implemented the ID3 algorithm from scratch using entropy and information gain, then compared the results with scikit-learn. |
| [`lab02-model-selection`](lab02-model-selection) | Model Selection | Manual hyperparameter tuning, GridSearchCV, and stratified k-fold cross-validation using Decision Trees, Logistic Regression, and KNN. |
| [`lab03-svm-kernels`](lab03-svm-kernels) | Support Vector Machines | Explored linear, polynomial, and RBF kernels on synthetic and real-world datasets. |
| [`lab04-naive-bayes`](lab04-naive-bayes) | Naive Bayes | Built Multinomial Naive Bayes from scratch, applied TF-IDF for text classification, and compared it with scikit-learn using a voting ensemble. |

---

## Time-Series Forecasting Project

The repository also contains a mini-project on forecasting daily Wikipedia page views.

Models explored include:

- K-Nearest Neighbors (baseline)
- LSTM
- Sequence-to-Sequence Causal CNN

Performance is evaluated using **SMAPE**. The notebook demonstrates the complete preprocessing pipeline and KNN implementation, while the deep learning models are trained through the project source code.

Project folder:

```
wikitraffic-forecasting/
```

---

## Running a Notebook

Install the required packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

Launch any notebook, for example:

```bash
jupyter notebook lab02-model-selection/grid_search_model_selection.ipynb
```

Most notebooks use datasets available through scikit-learn or generate synthetic datasets, making them easy to run without additional downloads.

---

## Technologies Used

- Python
- NumPy
- pandas
- scikit-learn
- TensorFlow
- Matplotlib
- Seaborn
- Jupyter Notebook
