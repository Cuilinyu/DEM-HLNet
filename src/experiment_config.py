"""Load dataset-specific settings on top of shared experiment defaults."""
import json

DEFAULT_SETTINGS = {
    "folds": 5,
    "random_state": 52,
    "transformer": {"width": 16, "heads": 2, "layers": 2},
    "gbdt": {"min_samples_leaf": 5, "early_stopping": False, "random_state": 52},
}


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    overrides = config["settings"]
    settings = {**DEFAULT_SETTINGS, **overrides}
    for model in ("transformer", "gbdt"):
        settings[model] = {**DEFAULT_SETTINGS[model], **overrides[model]}
    return {"test_fraction": 0.25, "cpu_threads": 4, **config, "settings": settings}
