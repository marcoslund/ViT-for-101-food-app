"""Conversion de metrics.json y training_history.csv al formato de MLflow."""

import pandas as pd

from vit_for_101_food_app import tracking


def test_split_values_separa_receta_metricas_y_tags():
    metrics = {
        "model_key": "deit",
        "receta": {"epochs": 20, "learning_rate": 5e-4},
        "test": {"accuracy": 0.73, "n": 25250},
        "latencia": {
            "gpu": None,
            "cpu": {"device": "cpu", "median_ms": 38.1},
            "cpu_int8": {"soportado": False, "motivo": "x"},
        },
    }
    params, numeric, tags = tracking.split_values(metrics)
    assert params == {"epochs": 20, "learning_rate": 5e-4}
    assert numeric == {"test/accuracy": 0.73, "test/n": 25250.0, "latencia/cpu/median_ms": 38.1}
    assert tags["model_key"] == "deit"
    assert tags["latencia/cpu_int8/soportado"] == "False"
    assert "latencia/gpu" not in numeric and "latencia/gpu" not in tags


def test_history_metrics_usa_los_tags_de_tensorboard():
    history = pd.DataFrame(
        [
            {"step": 200, "epoch": 0.5, "loss": 4.6, "learning_rate": 1e-4, "split": "train"},
            {"step": 400, "epoch": 1.0, "eval_loss": 1.5, "eval_f1_macro": 0.6, "split": "val"},
        ]
    )
    assert sorted(tracking.history_metrics(history)) == [
        ("eval/f1_macro", 0.6, 400),
        ("eval/loss", 1.5, 400),
        ("train/learning_rate", 1e-4, 200),
        ("train/loss", 4.6, 200),
    ]
