"""Run the fixed five-fold DEM-HLNet pointwise experiment for one dataset."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiment_config import load_config
from Transformer import TransformerRegressor
from plot_results import save_scatter


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["Longitude", "Latitude", "Slope", "Aspect", "Relief", "TRI", "TPI"]
TARGET = "Elevation_Error"


def load_dataset(path):
    frame = pd.read_excel(path).rename(columns={"Lontitute": "Longitude"})
    values = frame[FEATURES + [TARGET]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contains missing or nonfinite values.")
    return values[:, :-1], values[:, -1]


def build_model(settings):
    seed = settings["random_state"]
    return StackingRegressor(
        estimators=[
            ("transformer", make_pipeline(StandardScaler(), TransformerRegressor(
                **settings["transformer"], random_state=seed, device="cpu",
            ))),
            ("gbdt", make_pipeline(StandardScaler(), HistGradientBoostingRegressor(
                **settings["gbdt"],
            ))),
        ],
        final_estimator=ElasticNet(**settings["elasticnet"], max_iter=10000),
        cv=KFold(n_splits=settings["folds"], shuffle=True, random_state=seed),
        n_jobs=1,
    )


def pointwise_metrics(reference, prediction):
    residual = prediction - reference
    pearson_r = None
    if np.std(reference) > 0 and np.std(prediction) > 0:
        pearson_r = float(np.corrcoef(reference, prediction)[0, 1])
    return {
        "pearson_r": pearson_r,
        "mae_m": float(np.mean(np.abs(residual))),
        "rmse_m": float(np.sqrt(np.mean(residual ** 2))),
        "sd_m": float(np.std(residual, ddof=0)),
    }


def run(dataset, output, config):
    X, y = load_dataset(dataset)
    settings = config["settings"]
    development, test = train_test_split(
        np.arange(len(y)), test_size=config["test_fraction"],
        random_state=settings["random_state"],
    )
    torch.set_num_threads(config["cpu_threads"])
    model = build_model(settings)
    print(f"Training {dataset.stem} with {settings['folds']} folds.", flush=True)
    with threadpool_limits(limits=config["cpu_threads"], user_api="openmp"):
        model.fit(X[development], y[development])
        prediction = model.predict(X[test])
    metrics = {
        **pointwise_metrics(y[test], prediction),
        "development_samples": len(development),
        "test_samples": len(test),
        "settings": settings,
        "test_fraction": config["test_fraction"],
        "cpu_threads": config["cpu_threads"],
        "fusion_coefficients": model.final_estimator_.coef_.tolist(),
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "source_row": test + 2,
        "reference_error_m": y[test],
        "predicted_error_m": prediction,
        "prediction_residual_m": prediction - y[test],
    }).to_csv(output / "test_predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    joblib.dump(model, output / "model.joblib")
    save_scatter(prediction, y[test], metrics, output / "test_scatter.png", config["title"])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    datasets = sorted(path.stem for path in (ROOT / "configs").glob("*.json"))
    parser.add_argument("dataset", choices=datasets)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(ROOT / "configs" / f"{args.dataset}.json")
    output = args.output or ROOT / "results" / args.dataset
    metrics = run(ROOT / "datasets" / f"{args.dataset}.xlsx", output, config)
    print(json.dumps(metrics, indent=2))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
