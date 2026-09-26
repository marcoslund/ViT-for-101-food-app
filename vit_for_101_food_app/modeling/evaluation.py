"""Protocolo de evaluacion comun a todos los modelos del benchmark.

Vive en el paquete y no en cada notebook por la misma razon que la geometria de los
transforms: si MobileViT y ViT miden accuracy, FLOPs o latencia con codigo distinto, la
diferencia medida deja de ser una diferencia entre arquitecturas.

Tres piezas:
  - predicciones por imagen (``predictions_frame``), que se guardan a disco para poder
    comparar modelos sin reentrenar;
  - metricas sobre el subset del benchmark por tercil de dificultad
    (``benchmark_metrics``), que es lo que contesta la pregunta del proyecto;
  - costo arquitectonico: FLOPs (``count_flops``) y latencia (``measure_latency``).
"""

import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch

from vit_for_101_food_app.config import (
    BENCHMARK_MANIFEST,
    BENCHMARK_SUBSET,
    FOOD101_META_DIR,
    TRAIN_VAL_SPLIT,
)
from vit_for_101_food_app.preprocessing import splits

TERCILES = ("facil", "medio", "dificil")


def predictions_frame(
    frame: pd.DataFrame, logits: np.ndarray, id2label: dict[int, str]
) -> pd.DataFrame:
    """Una fila por imagen: que clase era, que predijo el modelo y si acerto.

    ``frame`` tiene que estar en el mismo orden que el dataset que produjo ``logits``
    (el de ``splits.load_split``, que es el que usa ``loaders.build_dataloaders``).
    """
    if len(frame) != len(logits):
        raise ValueError(f"{len(frame)} filas en frame pero {len(logits)} filas de logits")

    label2id = {v: k for k, v in id2label.items()}
    label_id = frame["class_dir"].map(label2id).to_numpy()
    if np.isnan(label_id.astype(float)).any():
        raise ValueError("frame tiene clases que no estan en id2label")

    top5 = np.argsort(logits, axis=-1)[:, ::-1][:, :5]
    pred_id = top5[:, 0]
    return pd.DataFrame(
        {
            "rel": frame["rel"].to_numpy(),
            "class_dir": frame["class_dir"].to_numpy(),
            "label_id": label_id.astype(int),
            "pred_id": pred_id,
            "pred_class": [id2label[int(i)] for i in pred_id],
            "correct": pred_id == label_id,
            "top5_correct": (top5 == label_id[:, None]).any(axis=1),
        }
    )


def split_metrics(preds: pd.DataFrame) -> dict:
    """Metricas de clasificacion sobre un conjunto de predicciones.

    El F1 macro se promedia solo sobre las clases presentes en ``label_id``: sobre un
    tercil (un tercio de las clases), promediar sobre las 101 meteria ceros por clases
    que no estan en la muestra.
    """
    y, p = preds["label_id"].to_numpy(), preds["pred_id"].to_numpy()
    clases = np.unique(y)
    return {
        "n": len(preds),
        "n_clases": len(clases),
        "accuracy": float(accuracy_score(y, p)),
        "top5_accuracy": float(preds["top5_correct"].mean()),
        "f1_macro": float(f1_score(y, p, labels=clases, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, p, labels=clases, average="weighted", zero_division=0)),
    }


def load_benchmark_subset(
    subset_path: Path = BENCHMARK_SUBSET, manifest_path: Path = BENCHMARK_MANIFEST
) -> pd.DataFrame:
    """Lee el subset del benchmark verificando su sha256 contra el manifiesto.

    Si el CSV cambio, las metricas dejarian de ser comparables con las de otros modelos
    sin que nada falle: mejor cortar aca.
    """
    esperado = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["csv_sha256"]
    real = hashlib.sha256(Path(subset_path).read_bytes()).hexdigest()
    if real != esperado:
        raise ValueError(
            f"{subset_path} no coincide con su manifiesto (sha256 {real} != {esperado})"
        )
    return pd.read_csv(subset_path)


def benchmark_metrics(preds: pd.DataFrame, subset: pd.DataFrame) -> pd.DataFrame:
    """Metricas sobre el subset del benchmark: global y por tercil de dificultad.

    ``preds`` son las predicciones sobre el test completo; el subset esta contenido en
    el. Falta alguna imagen del subset -> error, porque metricas sobre un subset
    incompleto no son comparables entre modelos.
    """
    unido = subset[["rel", "tercil"]].merge(preds, on="rel", how="left", validate="1:1")
    faltan = unido["pred_id"].isna().sum()
    if faltan:
        raise ValueError(f"{faltan} imagenes del subset no tienen prediccion")

    filas = [{"grupo": "subset", **split_metrics(unido)}]
    for tercil in TERCILES:
        filas.append({"grupo": tercil, **split_metrics(unido[unido["tercil"] == tercil])})
    return pd.DataFrame(filas).set_index("grupo")


def count_flops(model: torch.nn.Module, pixel_values: torch.Tensor) -> dict:
    """FLOPs de un forward, por imagen, medidos con el contador de torch.

    Se reportan FLOPs y MACs (= FLOPs / 2) porque la literatura mezcla las dos
    convenciones: el paper de MobileViT, por ejemplo, reporta MACs bajo el nombre FLOPs.
    """
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as contador:
        model(pixel_values=pixel_values)
    flops = contador.get_total_flops() / pixel_values.shape[0]
    return {"gflops": flops / 1e9, "gmacs": flops / 2e9}


def measure_latency(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    device: str | torch.device,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict:
    """Latencia de inferencia (ms por forward) en ``device``, con el batch dado.

    Mide cada corrida por separado y reporta mediana y p90, no el promedio: una sola
    interrupcion del sistema operativo mueve el promedio pero no la mediana.
    Deja el modelo en ``device``.
    """
    device = torch.device(device)
    model.eval().to(device)
    x = pixel_values.to(device)
    es_cuda = device.type == "cuda"

    tiempos = []
    with torch.no_grad():
        for i in range(n_warmup + n_runs):
            if es_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(pixel_values=x)
            if es_cuda:
                torch.cuda.synchronize(device)
            if i >= n_warmup:
                tiempos.append((time.perf_counter() - t0) * 1000)

    return {
        "device": torch.cuda.get_device_name(device) if es_cuda else "cpu",
        "batch_size": int(x.shape[0]),
        "cpu_threads": torch.get_num_threads(),
        "n_runs": n_runs,
        "median_ms": statistics.median(tiempos),
        "p90_ms": float(np.percentile(tiempos, 90)),
        "mean_ms": statistics.fmean(tiempos),
    }


def evaluate_test(
    trainer,
    test_dataset,
    id2label: dict[int, str],
    csv_path: Path = TRAIN_VAL_SPLIT,
    meta_dir: Path = FOOD101_META_DIR,
) -> tuple[pd.DataFrame, dict, float]:
    """Corre el modelo sobre test y arma predicciones + metricas globales.

    Devuelve ``(predicciones, metricas, segundos)``. El ``frame`` del split se lee con el
    mismo ``splits.load_split("test", ...)`` que uso el DataLoader, asi que queda en el
    orden en que ``trainer.predict`` produjo los logits (contrato de ``predictions_frame``).
    El caller decide si escribe el CSV: identico para el notebook y para el CLI headless.
    """
    t0 = time.time()
    output = trainer.predict(test_dataset)
    elapsed = time.time() - t0

    frame = splits.load_split("test", csv_path=csv_path, meta_dir=meta_dir)
    preds = predictions_frame(frame, output.predictions, id2label)
    return preds, split_metrics(preds), elapsed


def per_class_report(preds: pd.DataFrame, id2label: dict[int, str]) -> pd.DataFrame:
    """Precision/recall/F1 por clase, ordenado de peor a mejor F1.

    El orden es lo util para el informe: la cola de arriba son las clases donde el modelo
    falla, que es donde se juega la pregunta del benchmark.
    """
    class_names = [id2label[i] for i in range(len(id2label))]
    reporte = classification_report(
        preds["label_id"],
        preds["pred_id"],
        labels=list(range(len(id2label))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(reporte).T.loc[class_names].sort_values("f1-score")


def frequent_confusions(preds: pd.DataFrame) -> pd.DataFrame:
    """Pares (clase real -> clase predicha) mas frecuentes entre los errores."""
    errores = preds[~preds["correct"]]
    return (
        errores.groupby(["class_dir", "pred_class"])
        .size()
        .rename("n")
        .sort_values(ascending=False)
        .reset_index()
    )


def architectural_cost(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    device: str | torch.device,
) -> tuple[dict, dict]:
    """Costo arquitectonico de un modelo: ``(costo, latencia)``.

    ``costo`` junta parametros, tamanio en disco (fp32) y FLOPs por imagen; ``latencia``
    mide en GPU (si ``device`` es cuda) y siempre en CPU. Deja el modelo en ``device`` al
    salir: ``measure_latency`` lo mueve a CPU para medir ahi, y hay que devolverlo.
    """
    device = torch.device(device)
    n_params = sum(p.numel() for p in model.parameters())
    costo = {
        "params_m": n_params / 1e6,
        "size_mb_fp32": n_params * 4 / 1024**2,
        **count_flops(model, pixel_values.to(device)),
    }
    latencia = {
        "gpu": measure_latency(model, pixel_values, device) if device.type == "cuda" else None,
        "cpu": measure_latency(model, pixel_values, "cpu"),
    }
    model.to(device)
    return costo, latencia
