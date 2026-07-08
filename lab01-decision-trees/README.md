# Decision Trees from Scratch

Implementing the split criterion a decision tree runs on - entropy, weighted
attribute information, and information gain - and using it to build an ID3 tree
on the *Play Tennis* dataset, then checking the root split against scikit-learn.

## What's here

- `entropy`, `attribute_info`, `information_gain` written from first principles
- ID3 built recursively from those functions
- a bar chart of per-attribute information gain
- a cross-check that sklearn's entropy tree picks the same root split

## Run

```bash
jupyter notebook decision_trees_from_scratch.ipynb
```

Needs `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
