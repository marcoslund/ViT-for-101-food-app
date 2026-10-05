"""Comparacion entre modelos a partir de reports/results: carga, contrastes y brechas."""

import json

import numpy as np
import pandas as pd
import pytest

from vit_for_101_food_app.modeling import comparison

RECETA = {
    "epochs": 20,
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "batch_size": 16,
    "grad_accum_steps": 2,
    "early_stopping_patience": 3,
    "metric_for_best": "f1_macro",
    "seed": 42,
    "epochs_entrenadas": 20.0,
}


def _metrics(model_key, f1, params, **receta):
    return {
        "model_key": model_key,
        "checkpoint": f"org/{model_key}",
        "receta": {**RECETA, **receta},
        "tiempo_entrenamiento_s": 3600.0,
        "best_val_f1_macro": f1 - 0.05,
        "test": {"n": 6, "accuracy": f1, "top5_accuracy": 1.0, "f1_macro": f1},
        "costo": {
            "params_m": params,
            "gflops": params / 2,
            "size_mb_fp32": params * 4,
            "size_mb_int8": params,
        },
        "latencia": {
            "gpu": None,
            "cpu": {"median_ms": params * 10},
            "cpu_int8": {"soportado": True, "size_mb": params * 1.1, "median_ms": params * 5},
        },
        "entorno": {"torch": "x"},
    }


@pytest.fixture
def results_dir(tmp_path):
    rels = ["a/1", "a/2", "b/1", "b/2", "c/1", "c/2"]
    aciertos = {
        "vit": [True, True, True, False, True, False],
        "mobilevit": [True, True, True, True, False, False],
    }
    for m, f1, params in (("vit", 0.6, 86.0), ("mobilevit", 0.7, 5.0)):
        d = tmp_path / m
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps(_metrics(m, f1, params)))
        clases = [r[0] for r in rels]
        pred = [c if ok else ("c" if c != "c" else "a") for c, ok in zip(clases, aciertos[m])]
        pd.DataFrame(
            {"rel": rels, "class_dir": clases, "correct": aciertos[m], "pred_class": pred}
        ).to_csv(d / "predictions_test.csv", index=False)
        errores = pd.DataFrame({"class_dir": clases, "pred_class": pred, "ok": aciertos[m]}).query(
            "not ok"
        )
        errores.groupby(["class_dir", "pred_class"]).size().rename("n").reset_index().to_csv(
            d / "confusiones_test.csv", index=False
        )
        pd.DataFrame({"f1-score": {"a": 0.9, "b": 0.5 if m == "vit" else 0.8, "c": 0.4}}).to_csv(
            d / "report_por_clase_test.csv"
        )
    return tmp_path


@pytest.fixture
def difficulty():
    return pd.DataFrame(
        {
            "label": ["A", "B", "C"],
            "margen": [0.1, 0.0, -0.1],
            "tercil": ["facil", "medio", "dificil"],
            "vecino_mas_cercano": ["B", "C", "A"],
        },
        index=pd.Index(["a", "b", "c"], name="class_dir"),
    )


def test_available_models_sigue_el_orden_de_colores(results_dir):
    assert comparison.available_models(results_dir) == ["mobilevit", "vit"]


def test_check_comparable_acepta_distinta_cantidad_de_epocas(results_dir):
    metrics = comparison.load_metrics(["mobilevit", "vit"], results_dir)
    metrics["mobilevit"]["receta"]["epochs_entrenadas"] = 17.0
    tabla = comparison.check_comparable(metrics)
    assert tabla.loc["epochs_entrenadas"].tolist() == [17.0, 20.0]


def test_check_comparable_rechaza_recetas_distintas():
    metrics = {
        "vit": _metrics("vit", 0.6, 86.0, learning_rate=5e-5),
        "mobilevit": _metrics("mobilevit", 0.7, 5.0),
    }
    with pytest.raises(ValueError, match="learning_rate"):
        comparison.check_comparable(metrics)


def test_summary_table_separa_int8_cota_de_medido(results_dir):
    metrics = comparison.load_metrics(["mobilevit", "vit"], results_dir)
    tabla = comparison.summary_table(metrics)
    assert tabla.loc["mobilevit", "size_mb_int8_cota"] == pytest.approx(5.0)
    assert tabla.loc["mobilevit", "size_mb_int8_medido"] == pytest.approx(5.5)
    assert tabla.loc["vit", "horas_entrenamiento"] == pytest.approx(1.0)


def test_load_predictions_une_por_imagen(results_dir):
    preds = comparison.load_predictions(["mobilevit", "vit"], results_dir)
    assert list(preds.columns) == [
        "rel",
        "class_dir",
        "mobilevit",
        "pred_mobilevit",
        "vit",
        "pred_vit",
    ]
    assert preds.set_index("rel").loc["c/1", "pred_mobilevit"] == "a"
    assert len(preds) == 6


def test_load_predictions_rechaza_conjuntos_distintos(results_dir):
    p = pd.read_csv(results_dir / "vit" / "predictions_test.csv")
    p.iloc[:5].to_csv(results_dir / "vit" / "predictions_test.csv", index=False)
    with pytest.raises(ValueError, match="mismas imágenes"):
        comparison.load_predictions(["mobilevit", "vit"], results_dir)


def test_paired_difference_cuenta_solo_los_discordantes():
    a = [True, True, True, False, False]
    b = [True, False, False, True, False]
    res = comparison.paired_difference(a, b)
    assert (res["solo_a"], res["solo_b"]) == (2, 1)
    assert res["diff"] == pytest.approx(0.2)
    assert res["ic_bajo"] < res["diff"] < res["ic_alto"]
    # Solo la diferencia y su intervalo: ningun p-valor que leer.
    assert "p_mcnemar" not in res


def test_paired_difference_sin_discordantes_da_diferencia_cero():
    res = comparison.paired_difference([True, False], [True, False])
    assert res["diff"] == 0
    assert res["ic_bajo"] == res["ic_alto"] == 0


def test_pareto_front_descarta_los_dominados():
    summary = pd.DataFrame(
        {"costo": [1.0, 2.0, 3.0, 2.0], "f1_macro": [0.7, 0.8, 0.75, 0.8]},
        index=["barato", "mejor", "dominado", "empate"],
    )
    frente = comparison.pareto_front(summary, "costo")
    assert frente.to_dict() == {"barato": True, "mejor": True, "dominado": False, "empate": True}


def test_gap_by_tercil_compara_dentro_de_cada_tercil(results_dir, difficulty):
    preds = comparison.with_tercil(
        comparison.load_predictions(["mobilevit", "vit"], results_dir), difficulty
    )
    gaps = comparison.gap_by_tercil(preds, ["mobilevit", "vit"]).set_index("tercil")
    assert gaps["diff"].to_dict() == pytest.approx({"facil": 0.0, "medio": 0.5, "dificil": -0.5})
    assert set(gaps["modelo"]) == {"mobilevit"}


def test_with_tercil_rechaza_clases_sin_dificultad(results_dir, difficulty):
    preds = comparison.load_predictions(["mobilevit", "vit"], results_dir)
    with pytest.raises(ValueError, match="sin tercil"):
        comparison.with_tercil(preds, difficulty.drop(index="c"))


def test_per_class_gap_y_flatness(results_dir, difficulty):
    tabla = comparison.per_class_gap(["mobilevit", "vit"], difficulty, results_dir=results_dir)
    assert tabla["gap_mobilevit"].to_dict() == pytest.approx({"c": 0.0, "b": 0.3, "a": 0.0})
    res = comparison.flatness(tabla, "mobilevit")
    assert (res["clases_gana"], res["clases_pierde"]) == (1, 0)
    assert -1 <= res["spearman_rho"] <= 1


def test_convergence_detecta_una_corrida_que_seguia_mejorando():
    h = pd.DataFrame(
        {
            "epoch": np.arange(1, 6),
            "eval_f1_macro": [0.5, 0.6, 0.7, 0.72, 0.75],
            "eval_loss": [1.0, 0.9, 0.95, 1.1, 1.2],
        }
    )
    fila = comparison.convergence({"m": h}).loc["m"]
    assert fila["epoca_mejor"] == 5
    assert fila["mejora_ultimas"] == pytest.approx(0.15)
    assert fila["val_loss_min"] == pytest.approx(0.9)
    assert fila["epoca_val_loss_min"] == 2


def test_confusion_pairs_suma_los_modelos_y_ordena_por_total(results_dir):
    tabla = comparison.confusion_pairs(["mobilevit", "vit"], results_dir=results_dir)
    primera = tabla.iloc[0]
    assert (primera["clase_real"], primera["clase_predicha"]) == ("c", "a")
    assert (primera["mobilevit"], primera["vit"], primera["total"]) == (2, 1, 3)
    # un par que un modelo nunca confundio cuenta 0, no NaN
    par_bc = tabla[(tabla["clase_real"] == "b") & (tabla["clase_predicha"] == "c")].iloc[0]
    assert (par_bc["mobilevit"], par_bc["vit"]) == (0, 1)


def test_present_traduce_encabezados_modelos_y_terciles():
    tabla = pd.DataFrame(
        {"modelo": ["mobilevit"], "tercil": ["dificil"], "diff": [0.05], "gap_swin": [0.1]}
    )
    out = comparison.present(tabla)
    assert list(out.columns) == ["Modelo", "Tercil", "Diferencia", "Brecha Swin-T"]
    assert out.iloc[0].tolist()[:2] == ["MobileViT-S", "Difícil"]
    # la tabla original no cambia: los CSV se guardan con las claves
    assert list(tabla.columns) == ["modelo", "tercil", "diff", "gap_swin"]


def test_unordered_pairs_suma_los_dos_sentidos():
    conf = pd.DataFrame(
        {
            "clase_real": ["steak", "filet", "cake"],
            "clase_predicha": ["filet", "steak", "mousse"],
            "total": [5, 4, 6],
        }
    )
    pares = comparison.unordered_pairs(conf, top=2)
    assert pares.to_dict("records") == [
        {"clase_a": "filet", "clase_b": "steak", "total": 9},
        {"clase_a": "cake", "clase_b": "mousse", "total": 6},
    ]


def test_present_paired_pasa_a_puntos_y_nombra_a_y_b():
    tabla = pd.DataFrame([comparison.paired_difference([True, True, False], [True, False, False])])
    out = comparison.present_paired(tabla)
    assert out.loc[0, "Diferencia A − B (puntos)"] == pytest.approx(100 / 3)
    assert out.loc[0, "Acierta solo A"] == 1 and out.loc[0, "Acierta solo B"] == 0
    assert "p (McNemar)" not in out.columns
    assert out.loc[0, "Imágenes"] == 3


# --------------------------------------------------------------------------------------
# Varias recetas por arquitectura
# --------------------------------------------------------------------------------------


def _historia(f1s):
    filas = [
        {"epoch": i + 1, "split": "val", "eval_f1_macro": f, "eval_loss": 1 - f}
        for i, f in enumerate(f1s)
    ]
    return pd.DataFrame(filas)


@pytest.fixture
def results_recetas(results_dir):
    """``results_dir`` mas una segunda corrida de vit con otro learning rate."""
    d = results_dir / "vit-lr5e-5"
    d.mkdir()
    (d / "metrics.json").write_text(
        json.dumps(_metrics("vit", 0.8, 86.0, learning_rate=5e-5, epochs_entrenadas=8.0))
    )
    base = pd.read_csv(results_dir / "mobilevit" / "predictions_test.csv")
    base.to_csv(d / "predictions_test.csv", index=False)
    pd.read_csv(results_dir / "mobilevit" / "confusiones_test.csv").to_csv(
        d / "confusiones_test.csv", index=False
    )
    pd.read_csv(results_dir / "mobilevit" / "report_por_clase_test.csv", index_col=0).to_csv(
        d / "report_por_clase_test.csv"
    )
    _historia([0.5, 0.6]).to_csv(results_dir / "vit" / "training_history.csv", index=False)
    _historia([0.7, 0.8, 0.75]).to_csv(d / "training_history.csv", index=False)
    _historia([0.6, 0.7]).to_csv(results_dir / "mobilevit" / "training_history.csv", index=False)
    return results_dir


def test_load_metrics_acepta_un_dict_clave_a_directorio(results_recetas):
    metrics = comparison.load_metrics(
        {"vit": "vit-lr5e-5", "mobilevit": "mobilevit"}, results_recetas
    )
    assert list(metrics) == ["vit", "mobilevit"]
    assert metrics["vit"]["receta"]["learning_rate"] == 5e-5


def test_load_predictions_acepta_un_dict_clave_a_directorio(results_recetas):
    preds = comparison.load_predictions(
        {"vit": "vit-lr5e-5", "mobilevit": "mobilevit"}, results_recetas
    )
    assert {"vit", "pred_vit", "mobilevit", "pred_mobilevit"} <= set(preds.columns)
    # vit-lr5e-5 copia las predicciones de mobilevit: acierta lo mismo.
    assert preds["vit"].tolist() == preds["mobilevit"].tolist()


def test_funciones_que_leen_de_disco_aceptan_un_dict(results_recetas, difficulty):
    corridas = {"vit": "vit-lr5e-5", "mobilevit": "mobilevit"}
    por_clase = comparison.per_class_gap(corridas, difficulty, "vit", results_recetas)
    assert (por_clase["gap_mobilevit"] == 0).all()
    confusiones = comparison.confusion_pairs(corridas, results_dir=results_recetas)
    assert (confusiones["vit"] == confusiones["mobilevit"]).all()
    historias = comparison.load_histories(corridas, results_recetas)
    assert len(historias["vit"]) == 3


def test_check_comparable_permite_que_varien_los_factores_de_diseno(results_recetas):
    metrics = comparison.load_metrics(["vit", "vit-lr5e-5", "mobilevit"], results_recetas)
    tabla = comparison.check_comparable(metrics, design=("learning_rate",))
    assert tabla.loc["learning_rate", "vit-lr5e-5"] == 5e-5
    metrics["mobilevit"]["receta"]["seed"] = 7
    with pytest.raises(ValueError, match="seed"):
        comparison.check_comparable(metrics, design=("learning_rate",))


def test_run_table_una_fila_por_corrida_con_su_arquitectura_y_receta(results_recetas):
    metrics = comparison.load_metrics(["vit", "vit-lr5e-5", "mobilevit"], results_recetas)
    tabla = comparison.run_table(metrics)
    assert tabla.loc["vit-lr5e-5", "arquitectura"] == "vit"
    assert tabla.loc["vit-lr5e-5", "learning_rate"] == 5e-5
    assert tabla.loc["vit-lr5e-5", "epochs_entrenadas"] == 8.0
    assert tabla.loc["vit-lr5e-5", "mejor_val_f1"] == pytest.approx(0.75)
    assert tabla.loc["vit-lr5e-5", "f1_macro"] == pytest.approx(0.8)


def test_fmt_lr_sin_ceros_en_el_exponente():
    assert comparison.fmt_lr(5e-5) == "5e-5"
    assert comparison.fmt_lr(5e-4) == "5e-4"
    assert comparison.fmt_lr(1e-4) == "1e-4"


def test_run_label_nombra_arquitectura_y_receta(results_recetas):
    metrics = comparison.load_metrics(["vit", "vit-lr5e-5"], results_recetas)
    assert comparison.run_label("vit-lr5e-5", metrics) == "ViT-B/16 - lr 5e-5 - 20 ép."
    assert comparison.run_label("vit", metrics) == "ViT-B/16 - lr 5e-4 - 20 ép."


def test_best_runs_elige_por_f1_de_validacion_dentro_de_cada_arquitectura(results_recetas):
    metrics = comparison.load_metrics(["vit", "vit-lr5e-5", "mobilevit"], results_recetas)
    assert comparison.best_runs(metrics) == {"vit": "vit-lr5e-5", "mobilevit": "mobilevit"}


def test_plot_recipe_curves_un_panel_por_arquitectura(results_recetas):
    corridas = ["vit", "vit-lr5e-5", "mobilevit"]
    metrics = comparison.load_metrics(corridas, results_recetas)
    historias = comparison.load_histories(corridas, results_recetas)
    fig = comparison.plot_recipe_curves(historias, metrics)
    ejes = [ax for ax in fig.axes if ax.get_title(loc="left")]
    assert len(ejes) == 2
    lineas_por_eje = sorted(len(ax.get_lines()) for ax in ejes)
    assert lineas_por_eje == [1, 2]


def test_plot_recipe_curves_ordena_los_paneles_en_grilla(results_recetas):
    corridas = ["vit", "vit-lr5e-5", "mobilevit"]
    metrics = comparison.load_metrics(corridas, results_recetas)
    historias = comparison.load_histories(corridas, results_recetas)
    # Con dos columnas y dos arquitecturas, una fila; con una columna, uno debajo del otro.
    en_fila = [ax.get_position() for ax in comparison.plot_recipe_curves(historias, metrics).axes]
    assert en_fila[0].y0 == en_fila[1].y0
    apilados = [
        ax.get_position() for ax in comparison.plot_recipe_curves(historias, metrics, ncols=1).axes
    ]
    assert apilados[0].x0 == apilados[1].x0
    assert apilados[0].y0 > apilados[1].y0
