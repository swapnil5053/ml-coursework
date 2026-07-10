# SVM Kernels and Decision Boundaries

Comparing the linear, polynomial, and RBF kernels on a deliberately non-linear
problem (`make_moons`) and a real tabular set (Breast Cancer Wisconsin). The moons
half is the interesting one - you can see each kernel's boundary and where it
can't bend enough.

## What's here

- decision-boundary plots for all three kernels on the moons data
- the same kernels scored on a real dataset, with a full classification report
- a short read on when "more kernel" actually helps and when it doesn't

## Run

```bash
jupyter notebook svm_kernels_decision_boundaries.ipynb
```

Needs `numpy`, `matplotlib`, `scikit-learn`.
