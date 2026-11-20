# DEM-HLNet

Pointwise elevation-error prediction for AW3D30 and COPDEM in the Rocky Mountains (RM), Greater Khingan Mountains (GKM), and Great Dividing Range (GDR). Each of the six datasets has a fixed configuration corresponding to the DEM-HLNet row in Tables III–V of the manuscript.

## Run an experiment

Use Python 3.12 in a virtual environment. Install the PyTorch CPU build and the other dependencies:

```text
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python src/train.py RM_AW3D30
```

Replace `RM_AW3D30` with another dataset filename stem in `configs/`. Use `--output` to choose a different results folder. The default is `results/<dataset>/`. Training all six datasets can take time on a CPU.

The script randomly reserves 25% for testing (seed 52). The two base learners are a multi-head Transformer and histogram-based gradient boosting. Standardization is fitted inside each fold. A five-fold split of the development set generates out-of-fold predictions to fit the ElasticNet. Both base learners are fitted on the complete development set for prediction.

Each run saves `metrics.json`, `test_predictions.csv`, `model.joblib` and `test_scatter.png`. The CSV includes each sample's original Excel row number. Pearson r is calculated directly from the paired reference and predicted errors. MAE, RMSE and population-normalized SD are computed from the prediction residuals. Scatter axes display −20 to 20 m; all held-out samples are included in the fit and metrics.

`configs/` stores the exact selected settings, reference metrics and the expected number of test samples for each dataset. The reference metrics came from the recorded local runs with Python 3.12.14, PyTorch 2.8.0 CPU and the pinned dependencies. Numerical results can vary with platform and libraries.

## Scope

The six workbooks are the prepared pointwise data. This release covers the DEM-HLNet entries in Tables III–V. Raster preprocessing and correction, the six global DEM product comparisons, and the complete training workflows for the 11 baseline models are outside this release.
