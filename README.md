# DEM-HLNet

Pointwise elevation-error prediction for AW3D30 and COPDEM in the Rocky Mountains (RM), Greater Khingan Mountains (GKM), and Great Dividing Range (GDR).

## Install dependencies

Use Python 3.12 in a virtual environment. Install the PyTorch CPU build and the other dependencies:

```text
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

On the development set, five-fold cross-validation generates predictions from Transformer and histogram-based gradient boosting (GBDT) to train the ElasticNet combiner.

Each run saves `metrics.json`, `test_predictions.csv`, `model.joblib` and `test_scatter.png`. The CSV includes each sample's original Excel row number. Pearson r is calculated directly from the paired reference and predicted errors. MAE, RMSE and population-normalized SD are computed from the prediction residuals. Scatter axes display −20 to 20 m; all held-out samples are included in the fit and metrics.
