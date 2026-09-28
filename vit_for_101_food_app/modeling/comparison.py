"""Comparacion entre modelos a partir de lo que ya escribio cada corrida en ``reports/results``.

No entrena ni evalua nada: junta los ``metrics.json``, las predicciones por imagen y los
reportes por clase de cada modelo y contesta las dos preguntas del proyecto:

  1. cuanto rendimiento se pierde con una arquitectura mas liviana
     (``summary_table``, ``paired_difference``, ``pareto_front``);
  2. si esa diferencia es pareja entre clases o se concentra en las dificiles
     (``gap_by_tercil``, ``per_class_gap``, ``flatness``).

No importa torch: corre con las dependencias base del proyecto.
"""

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
from scipy import stats

from vit_for_101_food_app.config import (
    BENCHMARK_MANIFEST,
    BENCHMARK_SUBSET,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from vit_for_101_food_app.plots import GRID, INK, INK_MUTED

RESULTS_DIR = REPORTS_DIR / "results"
CLASS_DIFFICULTY = PROCESSED_DATA_DIR / "class_difficulty.csv"
TERCILES = ("facil", "medio", "dificil")
REFERENCE = "vit"

# Un color fijo por modelo (paleta categorica validada para daltonismo, en este orden):
# el color sigue al modelo en todas las figuras, nunca a su posicion en un ranking.
MODEL_COLORS = {
    "mobilevit": "#2a78d6",
    "vit": "#eb6834",
    "swin": "#1baf7a",
    "deit": "#eda100",
}

# Campos de la receta que tienen que coincidir para que la comparacion sea entre
# arquitecturas. ``epochs_entrenadas`` puede diferir: la corta el early stopping.
RECIPE_KEYS = (
    "epochs",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "early_stopping_patience",
    "metric_for_best",
    "seed",
)


# --------------------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------------------


def available_models(results_dir: Path = RESULTS_DIR) -> list[str]:
    """Modelos con ``metrics.json``, en el orden de ``MODEL_COLORS`` y despues alfabetico."""
    encontrados = {p.parent.name for p in Path(results_dir).glob("*/metrics.json")}
    orden = [m for m in MODEL_COLORS if m in encontrados]
    return orden + sorted(encontrados - set(orden))


def load_metrics(models: list[str], results_dir: Path = RESULTS_DIR) -> dict[str, dict]:
    return {
        m: json.loads((Path(results_dir) / m / "metrics.json").read_text(encoding="utf-8"))
        for m in models
    }


def check_comparable(metrics: dict[str, dict]) -> pd.DataFrame:
    """Receta y entorno de cada corrida, lado a lado. Error si la receta difiere.

    El entorno (GPU, versiones) no corta: se devuelve para que quede a la vista, porque
    las latencias solo son comparables si se midieron en la misma maquina.
    """
    filas = {}
    for m, d in metrics.items():
        receta = d["receta"]
        filas[m] = {
            **{k: receta.get(k) for k in RECIPE_KEYS},
            "batch_efectivo": receta["batch_size"] * receta["grad_accum_steps"],
            "epochs_entrenadas": receta.get("epochs_entrenadas"),
            "n_test": d["test"]["n"],
            **{f"entorno_{k}": v for k, v in d.get("entorno", {}).items()},
        }
    tabla = pd.DataFrame(filas)
    distintas = [
        k
        for k in (*RECIPE_KEYS, "batch_efectivo", "n_test")
        if tabla.loc[k].astype(str).nunique() > 1
    ]
    if distintas:
        raise ValueError(f"las corridas no usan la misma receta: difieren en {distintas}")
    return tabla


def load_predictions(models: list[str], results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Aciertos por imagen de todos los modelos en una sola tabla (una fila por imagen).

    Error si los modelos no predijeron exactamente las mismas imagenes: la comparacion
    pareada (``paired_difference``) necesita el mismo conjunto, imagen por imagen.
    """
    base = None
    for m in models:
        p = pd.read_csv(Path(results_dir) / m / "predictions_test.csv")
        p = p[["rel", "class_dir", "correct"]].rename(columns={"correct": m})
        if base is None:
            base = p
            continue
        if set(p["rel"]) != set(base["rel"]):
            raise ValueError(f"{m} no predijo las mismas imagenes que {models[0]}")
        base = base.merge(p.drop(columns="class_dir"), on="rel", validate="1:1")
    return base


def load_class_difficulty(path: Path = CLASS_DIFFICULTY) -> pd.DataFrame:
    return pd.read_csv(path).set_index("class_dir")


def load_benchmark_subset(
    subset_path: Path = BENCHMARK_SUBSET, manifest_path: Path = BENCHMARK_MANIFEST
) -> pd.DataFrame:
    """El mismo contrato que ``evaluation.load_benchmark_subset``, sin importar torch."""
    esperado = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["csv_sha256"]
    real = hashlib.sha256(Path(subset_path).read_bytes()).hexdigest()
    if real != esperado:
        raise ValueError(
            f"{subset_path} no coincide con su manifiesto (sha256 {real} != {esperado})"
        )
    return pd.read_csv(subset_path)


def load_histories(models: list[str], results_dir: Path = RESULTS_DIR) -> dict[str, pd.DataFrame]:
    """Filas de validacion de ``training_history.csv`` (una por epoca)."""
    out = {}
    for m in models:
        h = pd.read_csv(Path(results_dir) / m / "training_history.csv")
        out[m] = h[h["split"] == "val"].reset_index(drop=True)
    return out


# --------------------------------------------------------------------------------------
# Pregunta 1: cuanto se pierde
# --------------------------------------------------------------------------------------


def summary_table(metrics: dict[str, dict]) -> pd.DataFrame:
    """Desempenio y costo de cada modelo en una fila."""
    filas = {}
    for m, d in metrics.items():
        costo, lat = d["costo"], d["latencia"]
        int8 = lat.get("cpu_int8") or {}
        filas[m] = {
            "checkpoint": d["checkpoint"],
            "accuracy": d["test"]["accuracy"],
            "top5_accuracy": d["test"]["top5_accuracy"],
            "f1_macro": d["test"]["f1_macro"],
            "params_m": costo["params_m"],
            "gflops": costo["gflops"],
            "size_mb_fp32": costo["size_mb_fp32"],
            "size_mb_int8_cota": costo["size_mb_int8"],
            "size_mb_int8_medido": int8.get("size_mb") if int8.get("soportado") else None,
            "cpu_ms": (lat.get("cpu") or {}).get("median_ms"),
            "cpu_int8_ms": int8.get("median_ms") if int8.get("soportado") else None,
            "gpu_ms": (lat.get("gpu") or {}).get("median_ms"),
            "epochs_entrenadas": d["receta"].get("epochs_entrenadas"),
            "horas_entrenamiento": d["tiempo_entrenamiento_s"] / 3600,
        }
    return pd.DataFrame(filas).T.infer_objects()


def paired_difference(correct_a, correct_b, alpha: float = 0.05) -> dict:
    """Diferencia de accuracy ``a - b`` sobre las mismas imagenes, con IC y McNemar.

    Como los dos modelos se evaluan sobre las mismas imagenes, lo que informa es en
    cuantas acierta uno y el otro no (``solo_a``, ``solo_b``); las que aciertan o
    fallan los dos no aportan. El IC es el de Wald para proporciones pareadas y el
    p-valor, el de McNemar exacto (binomial sobre los pares discordantes).
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("los vectores de aciertos tienen que tener el mismo largo")
    n = len(a)
    solo_a = int((a & ~b).sum())
    solo_b = int((~a & b).sum())
    diff = (solo_a - solo_b) / n
    se = math.sqrt(max(solo_a + solo_b - (solo_a - solo_b) ** 2 / n, 0.0)) / n
    z = stats.norm.ppf(1 - alpha / 2)
    discordantes = solo_a + solo_b
    p = stats.binomtest(solo_a, discordantes).pvalue if discordantes else 1.0
    return {
        "n": n,
        "acc_a": float(a.mean()),
        "acc_b": float(b.mean()),
        "diff": diff,
        "ic_bajo": diff - z * se,
        "ic_alto": diff + z * se,
        "solo_a": solo_a,
        "solo_b": solo_b,
        "p_mcnemar": float(p),
    }


def differences_vs(
    preds: pd.DataFrame, models: list[str], reference: str = REFERENCE
) -> pd.DataFrame:
    """``paired_difference`` de cada modelo contra ``reference``, sobre todo ``preds``."""
    filas = {m: paired_difference(preds[m], preds[reference]) for m in models if m != reference}
    return pd.DataFrame(filas).T.infer_objects()


def pareto_front(summary: pd.DataFrame, cost_col: str, perf_col: str = "f1_macro") -> pd.Series:
    """True para los modelos que ningun otro supera a la vez en costo y en desempenio."""
    datos = summary[[cost_col, perf_col]].astype(float)
    frente = {}
    for m, fila in datos.iterrows():
        otros = datos.drop(index=m)
        domina = (otros[cost_col] <= fila[cost_col]) & (otros[perf_col] >= fila[perf_col])
        estricto = (otros[cost_col] < fila[cost_col]) | (otros[perf_col] > fila[perf_col])
        frente[m] = not (domina & estricto).any()
    return pd.Series(frente, name=f"pareto_{cost_col}")


# --------------------------------------------------------------------------------------
# Pregunta 2: donde se concentra la diferencia
# --------------------------------------------------------------------------------------


def with_tercil(preds: pd.DataFrame, difficulty: pd.DataFrame) -> pd.DataFrame:
    """Agrega a cada imagen el tercil de dificultad de su clase (ranking del EDA)."""
    out = preds.merge(difficulty[["tercil"]], left_on="class_dir", right_index=True, how="left")
    if out["tercil"].isna().any():
        raise ValueError("hay clases sin tercil en class_difficulty.csv")
    return out


def gap_by_tercil(
    preds: pd.DataFrame, models: list[str], reference: str = REFERENCE
) -> pd.DataFrame:
    """Diferencia pareada contra ``reference`` dentro de cada tercil (``preds`` con tercil)."""
    filas = []
    for tercil in TERCILES:
        parte = preds[preds["tercil"] == tercil]
        for m in models:
            if m == reference:
                continue
            filas.append(
                {"tercil": tercil, "modelo": m, **paired_difference(parte[m], parte[reference])}
            )
    return pd.DataFrame(filas)


def per_class_gap(
    models: list[str],
    difficulty: pd.DataFrame,
    reference: str = REFERENCE,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """F1 por clase de cada modelo y su brecha contra ``reference``, con la dificultad."""
    f1 = pd.DataFrame(
        {
            m: pd.read_csv(Path(results_dir) / m / "report_por_clase_test.csv", index_col=0)[
                "f1-score"
            ]
            for m in models
        }
    )
    f1 = f1.loc[f1.index.isin(difficulty.index)]
    tabla = f1.add_prefix("f1_")
    for m in models:
        if m != reference:
            tabla[f"gap_{m}"] = f1[m] - f1[reference]
    cols = ["label", "margen", "tercil", "vecino_mas_cercano"]
    return tabla.join(difficulty[cols]).sort_values("margen")


def flatness(per_class: pd.DataFrame, model: str) -> dict:
    """Si la brecha de ``model`` depende de la dificultad de la clase.

    Spearman entre la brecha de F1 por clase y el margen del EDA (margen bajo = clase
    dificil). Una brecha plana da rho cercano a 0; una brecha que se concentra en las
    clases dificiles da rho de signo definido.
    """
    gap = per_class[f"gap_{model}"]
    rho, p = stats.spearmanr(per_class["margen"], gap)
    pendiente = np.polyfit(per_class["margen"], gap, 1)[0]
    return {
        "modelo": model,
        "gap_medio": float(gap.mean()),
        "gap_desvio": float(gap.std()),
        "clases_gana": int((gap > 0).sum()),
        "clases_pierde": int((gap < 0).sum()),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "pendiente": float(pendiente),
    }


def convergence(histories: dict[str, pd.DataFrame], ultimas: int = 3) -> pd.DataFrame:
    """Si cada corrida habia dejado de mejorar cuando termino.

    ``mejora_ultimas`` es cuanto subio el F1 de validacion en las ultimas ``ultimas``
    epocas: si sigue siendo positiva al llegar al tope de epocas, el modelo no convergio
    y su resultado es una cota inferior de lo que la arquitectura puede dar.
    """
    filas = {}
    for m, h in histories.items():
        f1 = h["eval_f1_macro"].to_numpy()
        filas[m] = {
            "epocas": len(f1),
            "mejor_val_f1": float(f1.max()),
            "epoca_mejor": int(f1.argmax()) + 1,
            "val_f1_final": float(f1[-1]),
            "mejora_ultimas": float(f1[-1] - f1[-1 - ultimas]) if len(f1) > ultimas else None,
            "val_loss_min": float(h["eval_loss"].min()),
            "val_loss_final": float(h["eval_loss"].iloc[-1]),
        }
    return pd.DataFrame(filas).T.infer_objects()


# --------------------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------------------


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)


def _p_text(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _color(model: str) -> str:
    return MODEL_COLORS.get(model, INK_MUTED)


def plot_tradeoff(
    summary: pd.DataFrame,
    costs: dict[str, str],
    perf_col: str = "f1_macro",
) -> plt.Figure:
    """F1 contra cada eje de costo, un panel por eje (nunca dos escalas en un mismo eje).

    Cada punto lleva el nombre del modelo al lado: la identidad no depende solo del
    color. La linea punteada une el frente de Pareto de ese eje.
    """
    fig, axes = plt.subplots(1, len(costs), figsize=(4.2 * len(costs), 4), sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")
    for ax, (col, etiqueta) in zip(axes, costs.items()):
        datos = summary[[col, perf_col]].dropna().astype(float)
        frente = pareto_front(datos, col, perf_col)
        pf = datos[frente].sort_values(col)
        ax.plot(pf[col], pf[perf_col], color=INK_MUTED, linewidth=1, linestyle=":", zorder=1)
        for m, fila in datos.iterrows():
            ax.scatter(
                fila[col],
                fila[perf_col],
                s=70,
                color=_color(m),
                edgecolor="white",
                linewidth=2,
                zorder=2,
            )
            ax.annotate(
                m,
                (fila[col], fila[perf_col]),
                xytext=(7, -3),
                textcoords="offset points",
                fontsize=9,
                color=INK,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.margins(x=0.15)
        _style(
            ax, etiqueta, f"{etiqueta} (escala log)", "F1 macro (test)" if ax is axes[0] else ""
        )
    fig.suptitle(
        "Desempenio contra costo", x=0.02, ha="left", fontsize=12, color=INK, fontweight="bold"
    )
    fig.tight_layout()
    return fig


def plot_gap_by_tercil(gaps: pd.DataFrame, reference: str = REFERENCE) -> plt.Figure:
    """Diferencia de accuracy contra la referencia por tercil, con su IC del 95 %."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    fig.patch.set_facecolor("white")
    modelos = list(dict.fromkeys(gaps["modelo"]))
    x = np.arange(len(TERCILES))
    ancho = 0.18
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    for i, m in enumerate(modelos):
        parte = gaps[gaps["modelo"] == m].set_index("tercil").loc[list(TERCILES)]
        xs = x + (i - (len(modelos) - 1) / 2) * ancho
        y = parte["diff"] * 100
        err = np.vstack([y - parte["ic_bajo"] * 100, parte["ic_alto"] * 100 - y])
        ax.errorbar(
            xs, y, yerr=err, fmt="none", ecolor=_color(m), elinewidth=2, capsize=0, zorder=1
        )
        ax.plot(xs, y, color=_color(m), linewidth=1, alpha=0.5, zorder=1)
        ax.scatter(xs, y, s=64, color=_color(m), edgecolor="white", linewidth=2, zorder=2, label=m)
        ax.annotate(
            m,
            (xs[-1], y.iloc[-1]),
            xytext=(8, -3),
            textcoords="offset points",
            fontsize=8.5,
            color=INK,
        )
    ax.set_xticks(x, list(TERCILES))
    ax.set_xlim(-0.5, len(TERCILES) - 0.2)
    _style(
        ax,
        f"Diferencia de accuracy contra {reference}, por tercil (IC 95 %)",
        "tercil de dificultad de la clase (EDA)",
        "puntos porcentuales",
    )
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    fig.suptitle(
        "Donde se concentra la brecha",
        x=0.02,
        ha="left",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_class_gap(
    per_class: pd.DataFrame, model: str, reference: str = REFERENCE, n_etiquetas: int = 4
) -> plt.Figure:
    """Brecha de F1 por clase contra el margen de dificultad del EDA."""
    gap = per_class[f"gap_{model}"] * 100
    fig, ax = plt.subplots(figsize=(7, 4.4))
    fig.patch.set_facecolor("white")
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    ax.scatter(
        per_class["margen"],
        gap,
        s=36,
        color=_color(model),
        edgecolor="white",
        linewidth=1.5,
        zorder=2,
    )
    pendiente, ordenada = np.polyfit(per_class["margen"], gap, 1)
    xs = np.linspace(per_class["margen"].min(), per_class["margen"].max(), 50)
    ax.plot(xs, pendiente * xs + ordenada, color=INK_MUTED, linewidth=1.2, linestyle="--")
    extremos = pd.concat([gap.nsmallest(n_etiquetas), gap.nlargest(n_etiquetas)])
    for clase, valor in extremos.items():
        ax.annotate(
            clase,
            (per_class.loc[clase, "margen"], valor),
            xytext=(5, 2),
            textcoords="offset points",
            fontsize=7.5,
            color=INK_MUTED,
        )
    res = flatness(per_class, model)
    ax.text(
        0.99,
        0.03,
        f"Spearman rho = {res['spearman_rho']:.2f} ({_p_text(res['spearman_p'])})",
        transform=ax.transAxes,
        ha="right",
        fontsize=8.5,
        color=INK,
    )
    _style(
        ax,
        f"F1 de {model} menos F1 de {reference}, por clase",
        "margen de la clase en el EDA (izquierda = mas dificil)",
        "puntos de F1",
    )
    fig.suptitle(
        f"{model}: la brecha segun la dificultad de la clase",
        x=0.02,
        ha="left",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_validation_curves(histories: dict[str, pd.DataFrame]) -> plt.Figure:
    """F1 macro de validacion por epoca, todos los modelos juntos."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    fig.patch.set_facecolor("white")
    for m, h in histories.items():
        ax.plot(
            h["epoch"],
            h["eval_f1_macro"],
            color=_color(m),
            linewidth=2,
            marker="o",
            markersize=3.5,
            label=m,
        )
    # Etiquetas al final de cada curva, separadas para que no se pisen si terminan parejas.
    x_fin = max(h["epoch"].iloc[-1] for h in histories.values())
    finales = sorted(
        ((h["eval_f1_macro"].iloc[-1], m) for m, h in histories.items()), reverse=True
    )
    y_prev = None
    for valor, m in finales:
        y = valor if y_prev is None else min(valor, y_prev - 0.018)
        ax.annotate(f"{m} {valor:.3f}", (x_fin + 0.6, y), va="center", fontsize=8.5, color=INK)
        y_prev = y
    ax.set_xlim(right=x_fin + 5)
    _style(ax, "F1 macro de validacion por epoca", "epoca", "F1 macro")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.suptitle("Convergencia", x=0.02, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    return fig
