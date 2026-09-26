"""Protocolo de evaluacion comun: predicciones, metricas por tercil, FLOPs y latencia."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

import torch

from vit_for_101_food_app.modeling import evaluation

ID2LABEL = {0: "a", 1: "b", 2: "c"}


def _logits_que_predicen(preds: list[int]) -> np.ndarray:
    logits = np.zeros((len(preds), len(ID2LABEL)), dtype=np.float32)
    logits[np.arange(len(preds)), preds] = 1.0
    return logits


@pytest.fixture
def frame():
    return pd.DataFrame(
        {"rel": ["a/1", "a/2", "b/1", "b/2", "c/1", "c/2"], "class_dir": list("aabbcc")}
    )


def test_predictions_frame_marca_aciertos_y_top5(frame):
    preds = evaluation.predictions_frame(frame, _logits_que_predicen([0, 1, 1, 1, 2, 0]), ID2LABEL)
    assert preds["correct"].tolist() == [True, False, True, True, True, False]
    assert preds["pred_class"].tolist() == list("abbbca")
    # con 3 clases, la clase correcta siempre esta en el top-5
    assert preds["top5_correct"].all()


def test_predictions_frame_rechaza_largos_distintos(frame):
    with pytest.raises(ValueError, match="filas"):
        evaluation.predictions_frame(frame, _logits_que_predicen([0]), ID2LABEL)


def test_f1_macro_se_promedia_solo_sobre_las_clases_presentes(frame):
    preds = evaluation.predictions_frame(frame, _logits_que_predicen([0, 0, 1, 1, 2, 2]), ID2LABEL)
    # solo las filas de la clase "a": acierto perfecto -> F1 macro 1.0, no 1/3
    m = evaluation.split_metrics(preds[preds["class_dir"] == "a"])
    assert m["n_clases"] == 1
    assert m["f1_macro"] == pytest.approx(1.0)


def test_benchmark_metrics_por_tercil(frame):
    preds = evaluation.predictions_frame(frame, _logits_que_predicen([0, 0, 1, 0, 2, 0]), ID2LABEL)
    subset = pd.DataFrame(
        {"rel": ["a/1", "b/1", "b/2", "c/1"], "tercil": ["facil", "medio", "medio", "dificil"]}
    )
    tabla = evaluation.benchmark_metrics(preds, subset)
    assert tabla.loc["subset", "n"] == 4
    assert tabla.loc["facil", "accuracy"] == pytest.approx(1.0)
    assert tabla.loc["medio", "accuracy"] == pytest.approx(0.5)
    assert tabla.loc["dificil", "accuracy"] == pytest.approx(1.0)


def test_benchmark_metrics_exige_todas_las_imagenes_del_subset(frame):
    preds = evaluation.predictions_frame(frame, _logits_que_predicen([0, 0, 1, 1, 2, 2]), ID2LABEL)
    subset = pd.DataFrame({"rel": ["a/1", "z/9"], "tercil": ["facil", "facil"]})
    with pytest.raises(ValueError, match="no tienen prediccion"):
        evaluation.benchmark_metrics(preds, subset)


def test_load_benchmark_subset_verifica_el_hash(tmp_path):
    csv = tmp_path / "subset.csv"
    csv.write_text("rel,tercil\na/1,facil\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    sha = hashlib.sha256(csv.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({"csv_sha256": sha}), encoding="utf-8")
    assert len(evaluation.load_benchmark_subset(csv, manifest)) == 1

    csv.write_text("rel,tercil\na/1,dificil\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        evaluation.load_benchmark_subset(csv, manifest)


class _Lineal(torch.nn.Module):
    """Modelo minimo con la firma de HuggingFace: forward(pixel_values=...)."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 4, bias=False)

    def forward(self, pixel_values):
        return self.fc(pixel_values)


def test_count_flops_es_por_imagen():
    modelo = _Lineal()
    # un Linear 8->4 sin bias hace 8*4 multiply-adds = 64 FLOPs por imagen,
    # independientemente del tamanio del batch
    for batch in (1, 5):
        costo = evaluation.count_flops(modelo, torch.zeros(batch, 8))
        assert costo["gflops"] * 1e9 == pytest.approx(64)
        assert costo["gmacs"] * 1e9 == pytest.approx(32)


def test_measure_latency_en_cpu():
    lat = evaluation.measure_latency(_Lineal(), torch.zeros(1, 8), "cpu", n_warmup=2, n_runs=5)
    assert lat["device"] == "cpu"
    assert lat["batch_size"] == 1
    assert lat["n_runs"] == 5
    assert 0 < lat["median_ms"] <= lat["p90_ms"]


def test_model_size_estimates_escala_con_la_precision():
    # 1 M de parametros: 4 MB en fp32, la mitad en fp16, un cuarto en int8
    est = evaluation.model_size_estimates(1024**2)
    assert est["size_mb_fp32"] == pytest.approx(4.0)
    assert est["size_mb_fp16"] == pytest.approx(2.0)
    assert est["size_mb_int8"] == pytest.approx(1.0)


def test_measure_quantized_cpu_reduce_tamanio_y_mide_latencia():
    q = evaluation.measure_quantized_cpu(_Lineal(), torch.zeros(1, 8), n_warmup=2, n_runs=5)
    if not q["soportado"]:
        pytest.skip(f"cuantizacion no soportada en este entorno: {q.get('motivo')}")
    assert q["dtype"] == "qint8"
    assert q["size_mb"] > 0
    assert 0 < q["median_ms"] <= q["p90_ms"]
    assert q["n_runs"] == 5


def test_measure_quantized_cpu_no_rompe_ante_fallos(monkeypatch):
    def explota(*_args, **_kwargs):
        raise RuntimeError("sin soporte de qint8")

    monkeypatch.setattr(torch.ao.quantization, "quantize_dynamic", explota)
    q = evaluation.measure_quantized_cpu(_Lineal(), torch.zeros(1, 8))
    assert q["soportado"] is False
    assert "sin soporte de qint8" in q["motivo"]
