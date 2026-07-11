# Wikipedia Traffic Forecasting

Forecasting daily page views for Wikipedia articles, and comparing three very
different approaches to the same sequence-prediction problem:

- **KNN** - a nearest-neighbours regressor on lag/calendar features. The baseline.
- **LSTM** - a recurrent network that reads the recent history as a sequence.
- **Seq2Seq CNN** - a causal 1D-convolutional encoder/decoder.

Everything is scored with **SMAPE** (symmetric MAPE), which is the standard
metric for this kind of traffic data because it handles the huge range of view
counts without letting the biggest pages dominate.

## Why these three

They sit at three points on the effort/assumption curve. KNN makes no sequence
assumption at all and is nearly free to fit - it sets the bar. The LSTM models the
series as an ordered sequence. The causal CNN keeps the sequence view but trades
recurrence for dilated convolutions, which train faster and still respect time
order. Comparing them shows how much the sequence structure is actually worth on
this data.

## Layout

```
src/
  data_preprocessing.py   log transform, calendar + lag + rolling features, splits
  evaluation.py           SMAPE and the metrics table
  knn_model.py            KNN regressor + a distance-based anomaly flag
  lstm_model.py           LSTM sequence model
  cnn_seq2seq.py          causal seq2seq CNN
  main.py                 runs all three and writes a comparison
notebooks/
  knn_baseline_walkthrough.ipynb   the data pipeline + KNN baseline, end to end
data/                     validation scores; raw traffic is synthesised if absent
```

## Running it

```bash
pip install -r requirements.txt
python src/main.py          # trains and compares all three models
```

For just the baseline and the feature pipeline, open
`notebooks/knn_baseline_walkthrough.ipynb` - it runs in seconds on synthesised
data and needs only NumPy / pandas / scikit-learn / matplotlib.

## Notes

If you don't drop the real Wikipedia traffic file into `data/`, the preprocessor
generates a synthetic series with the same shape so the code stays runnable. The
deep models need TensorFlow (see `requirements.txt`); the KNN baseline and the
notebook do not.
