"""Comparación entre modelos a partir de lo que ya escribió cada corrida en ``reports/results``.

No entrena ni evalúa nada: junta los ``metrics.json``, las predicciones por imagen, los
reportes por clase y las confusiones de cada modelo, y produce las tablas y figuras que
usa el informe:

  - resultados globales y costo (``summary_table``, ``pareto_front``);
  - diferencias pareadas entre modelos (``paired_difference``, ``differences_vs``);
  - rendimiento según la dificultad de la clase (``gap_by_tercil``, ``per_class_gap``,
    ``flatness``);
  - confusiones compartidas (``confusion_pairs``) y convergencia (``convergence``);
  - varias recetas por arquitectura (``run_table``, ``best_runs``, ``plot_recipe_curves``).

Las funciones que leen de disco reciben los modelos como lista (la clave es el nombre del
directorio en ``reports/results``) o como dict ``clave -> directorio``, para comparar con
las claves de arquitectura de siempre corridas que viven en otro directorio (por ejemplo
``{"vit": "vit-lr5e-5"}``).

No importa torch: corre con las dependencias base del proyecto.

Las claves de las tablas son identificadores (``snake_case``, sin tildes) para que los CSV
se puedan releer desde código; ``present`` las traduce a encabezados legibles.
"""

import hashlib
import json
import math
from pathlib import Path

from matplotlib.lines import Line2D
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

# Un color fijo por modelo (paleta categórica validada para daltonismo, en este orden):
# el color sigue al modelo en todas las figuras, nunca a su posición en un ranking.
MODEL_COLORS = {
    "mobilevit": "#2a78d6",
    "vit": "#eb6834",
    "swin": "#1baf7a",
    "deit": "#eda100",
}

MODEL_NAMES = {
    "mobilevit": "MobileViT-S",
    "vit": "ViT-B/16",
    "swin": "Swin-T",
    "deit": "DeiT-Ti",
}

TERCIL_NAMES = {"facil": "Fácil", "medio": "Medio", "dificil": "Difícil", "subset": "Subset"}

# Encabezados legibles para mostrar las tablas en el notebook y en el informe.
COLUMN_LABELS = {
    # resultados y costo
    "checkpoint": "Checkpoint",
    "accuracy": "Accuracy",
    "top5_accuracy": "Accuracy top-5",
    "f1_macro": "F1 macro",
    "params_m": "Parámetros (M)",
    "gflops": "GFLOPs",
    "size_mb_fp32": "Tamaño fp32 (MB)",
    "size_mb_int8_cota": "Tamaño int8, cota teórica (MB)",
    "size_mb_int8_medido": "Tamaño int8, medido (MB)",
    "cpu_ms": "Latencia CPU fp32 (ms)",
    "cpu_int8_ms": "Latencia CPU int8 (ms)",
    "gpu_ms": "Latencia GPU (ms)",
    "epochs_entrenadas": "Épocas entrenadas",
    "horas_entrenamiento": "Horas de entrenamiento",
    # receta
    "arquitectura": "Arquitectura",
    "corrida": "Corrida",
    "epochs": "Épocas (tope)",
    "learning_rate": "Learning rate",
    "weight_decay": "Weight decay",
    "warmup_ratio": "Warmup",
    "early_stopping_patience": "Paciencia (early stopping)",
    "metric_for_best": "Métrica de selección",
    "seed": "Semilla",
    "batch_efectivo": "Batch efectivo",
    "n_test": "Imágenes de test",
    "entorno_torch": "PyTorch",
    "entorno_transformers": "Transformers",
    "entorno_gpu": "GPU",
    # diferencias pareadas
    "n": "Imágenes",
    "acc_a": "Accuracy del modelo",
    "acc_b": "Accuracy de la referencia",
    "diff": "Diferencia",
    "ic_bajo": "IC 95 % inferior",
    "ic_alto": "IC 95 % superior",
    "solo_a": "Acierta solo el modelo",
    "solo_b": "Acierta solo la referencia",
    "tercil": "Tercil",
    "modelo": "Modelo",
    # por clase
    "label": "Clase",
    "margen": "Margen (EDA)",
    "vecino_mas_cercano": "Vecina más confundible",
    "gap_medio": "Brecha media",
    "gap_desvio": "Desvío de la brecha",
    "clases_gana": "Clases en que gana",
    "clases_pierde": "Clases en que pierde",
    "spearman_rho": "Spearman ρ",
    "spearman_p": "p (Spearman)",
    "pendiente": "Pendiente",
    # confusiones
    "clase_real": "Clase real",
    "clase_predicha": "Clase predicha",
    "total": "Total",
    # convergencia
    "epocas": "Épocas",
    "mejor_val_f1": "Mejor F1 de validación",
    "epoca_mejor": "Época del mejor F1",
    "val_f1_final": "F1 de validación final",
    "mejora_ultimas": "Mejora en las últimas 3 épocas",
    "val_loss_min": "Pérdida de validación mínima",
    "epoca_val_loss_min": "Época de la pérdida mínima",
    "val_loss_final": "Pérdida de validación final",
}

# Campos de la receta que tienen que coincidir para que la comparación sea entre
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


def model_name(model: str) -> str:
    return MODEL_NAMES.get(model, model)


def present(table: pd.DataFrame) -> pd.DataFrame:
    """Copia de ``table`` con encabezados, modelos y terciles en nombres legibles.

    Solo para mostrar: los CSV se guardan con las claves originales.
    """
    nombres = {**MODEL_NAMES, **TERCIL_NAMES}

    def traducir(valor):
        if isinstance(valor, str):
            if valor in nombres:
                return nombres[valor]
            for prefijo in ("f1_", "gap_", "pred_"):
                if valor.startswith(prefijo) and valor[len(prefijo) :] in MODEL_NAMES:
                    modelo = model_name(valor[len(prefijo) :])
                    return {"f1_": f"F1 {modelo}", "gap_": f"Brecha {modelo}"}.get(
                        prefijo, f"Predicción {modelo}"
                    )
            return COLUMN_LABELS.get(valor, valor)
        return valor

    out = table.rename(index=traducir, columns=traducir)
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].map(lambda v: nombres.get(v, v) if isinstance(v, str) else v)
    return out


def present_paired(table: pd.DataFrame) -> pd.DataFrame:
    """Tabla de diferencias pareadas lista para leer.

    Cada fila compara un modelo A contra un modelo B (``paired_difference(a, b)``): las
    accuracies van en porcentaje y la diferencia A − B y su IC, en puntos porcentuales.
    """
    out = table.copy()
    for col in ("acc_a", "acc_b", "diff", "ic_bajo", "ic_alto"):
        if col in out:
            out[col] = out[col] * 100
    for col in ("n", "solo_a", "solo_b"):
        if col in out:
            out[col] = out[col].astype(int)
    etiquetas = {
        "acc_a": "Accuracy de A (%)",
        "acc_b": "Accuracy de B (%)",
        "diff": "Diferencia A − B (puntos)",
        "ic_bajo": "IC 95 %, desde (puntos)",
        "ic_alto": "IC 95 %, hasta (puntos)",
        "solo_a": "Acierta solo A",
        "solo_b": "Acierta solo B",
    }
    return present(out.rename(columns=etiquetas))


# --------------------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------------------


def _dirs(models) -> dict[str, str]:
    """``models`` como lista (clave == directorio) o como dict ``clave -> directorio``."""
    if isinstance(models, dict):
        return dict(models)
    return {m: m for m in models}


def available_models(results_dir: Path = RESULTS_DIR) -> list[str]:
    """Modelos con ``metrics.json``, en el orden de ``MODEL_COLORS`` y después alfabético."""
    encontrados = {p.parent.name for p in Path(results_dir).glob("*/metrics.json")}
    orden = [m for m in MODEL_COLORS if m in encontrados]
    return orden + sorted(encontrados - set(orden))


def load_metrics(models, results_dir: Path = RESULTS_DIR) -> dict[str, dict]:
    return {
        m: json.loads((Path(results_dir) / d / "metrics.json").read_text(encoding="utf-8"))
        for m, d in _dirs(models).items()
    }


def check_comparable(metrics: dict[str, dict], design: tuple[str, ...] = ()) -> pd.DataFrame:
    """Receta y entorno de cada corrida, lado a lado. Error si la receta difiere.

    ``design`` son los campos de la receta que se variaron a propósito (por ejemplo
    ``("learning_rate", "epochs")``): se muestran pero no cortan. Todo lo demás tiene que
    coincidir para que las diferencias sean entre arquitecturas o entre esos factores.

    El entorno (GPU, versiones) no corta: se devuelve para que quede a la vista, porque
    las latencias solo son comparables si se midieron en la misma máquina.
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
        if k not in design and tabla.loc[k].astype(str).nunique() > 1
    ]
    if distintas:
        raise ValueError(f"las corridas no usan la misma receta: difieren en {distintas}")
    return tabla


def load_predictions(models, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Predicciones por imagen de todos los modelos en una sola tabla.

    Una fila por imagen: la columna ``<modelo>`` dice si acertó y ``pred_<modelo>``, qué
    clase predijo. Error si los modelos no predijeron exactamente las mismas imágenes:
    la comparación pareada (``paired_difference``) necesita el mismo conjunto.
    """
    base = None
    dirs = _dirs(models)
    for m, d in dirs.items():
        p = pd.read_csv(Path(results_dir) / d / "predictions_test.csv")
        p = p[["rel", "class_dir", "correct", "pred_class"]].rename(
            columns={"correct": m, "pred_class": f"pred_{m}"}
        )
        if base is None:
            base = p
            continue
        if set(p["rel"]) != set(base["rel"]):
            raise ValueError(f"{m} no predijo las mismas imágenes que {next(iter(dirs))}")
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


def load_histories(models, results_dir: Path = RESULTS_DIR) -> dict[str, pd.DataFrame]:
    """Filas de validación de ``training_history.csv`` (una por época)."""
    out = {}
    for m, d in _dirs(models).items():
        h = pd.read_csv(Path(results_dir) / d / "training_history.csv")
        out[m] = h[h["split"] == "val"].reset_index(drop=True)
    return out


# --------------------------------------------------------------------------------------
# Resultados globales y diferencias entre modelos
# --------------------------------------------------------------------------------------


def summary_table(metrics: dict[str, dict]) -> pd.DataFrame:
    """Desempeño y costo de cada modelo en una fila."""
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
    """Diferencia de accuracy ``a - b`` sobre las mismas imágenes, con su IC del 95 %.

    Como los dos modelos se evalúan sobre las mismas imágenes, lo que informa es en
    cuántas acierta uno y el otro no (``solo_a``, ``solo_b``); las que aciertan o
    fallan los dos no aportan. El IC es el de Wald para proporciones pareadas: si no
    incluye al cero, la diferencia no se explica por la muestra de imágenes.
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
    return {
        "n": n,
        "acc_a": float(a.mean()),
        "acc_b": float(b.mean()),
        "diff": diff,
        "ic_bajo": diff - z * se,
        "ic_alto": diff + z * se,
        "solo_a": solo_a,
        "solo_b": solo_b,
    }


def differences_vs(
    preds: pd.DataFrame, models: list[str], reference: str = REFERENCE
) -> pd.DataFrame:
    """``paired_difference`` de cada modelo contra ``reference``, sobre todo ``preds``."""
    filas = {m: paired_difference(preds[m], preds[reference]) for m in models if m != reference}
    return pd.DataFrame(filas).T.infer_objects()


def pareto_front(summary: pd.DataFrame, cost_col: str, perf_col: str = "f1_macro") -> pd.Series:
    """True para los modelos que ningún otro supera a la vez en costo y en desempeño."""
    datos = summary[[cost_col, perf_col]].astype(float)
    frente = {}
    for m, fila in datos.iterrows():
        otros = datos.drop(index=m)
        domina = (otros[cost_col] <= fila[cost_col]) & (otros[perf_col] >= fila[perf_col])
        estricto = (otros[cost_col] < fila[cost_col]) | (otros[perf_col] > fila[perf_col])
        frente[m] = not (domina & estricto).any()
    return pd.Series(frente, name=f"pareto_{cost_col}")


# --------------------------------------------------------------------------------------
# Rendimiento según la dificultad de la clase
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
    models,
    difficulty: pd.DataFrame,
    reference: str = REFERENCE,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """F1 por clase de cada modelo y su brecha contra ``reference``, con la dificultad."""
    dirs = _dirs(models)
    models = list(dirs)
    f1 = pd.DataFrame(
        {
            m: pd.read_csv(Path(results_dir) / d / "report_por_clase_test.csv", index_col=0)[
                "f1-score"
            ]
            for m, d in dirs.items()
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
    difícil). Una brecha pareja da ρ cercano a 0; una brecha que se concentra en las
    clases difíciles da ρ negativo.
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


def confusion_pairs(models, top: int = 12, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Pares (clase real → clase predicha) más frecuentes, con el conteo de cada modelo.

    Se ordenan por el total sobre todos los modelos: arriba quedan las confusiones que
    comparten las arquitecturas, que hablan más del dataset que de un modelo.
    """
    tabla = None
    dirs = _dirs(models)
    models = list(dirs)
    for m, d in dirs.items():
        c = pd.read_csv(Path(results_dir) / d / "confusiones_test.csv").rename(
            columns={"class_dir": "clase_real", "pred_class": "clase_predicha", "n": m}
        )
        tabla = (
            c
            if tabla is None
            else tabla.merge(c, on=["clase_real", "clase_predicha"], how="outer")
        )
    tabla[models] = tabla[models].fillna(0).astype(int)
    tabla["total"] = tabla[models].sum(axis=1)
    return (
        tabla.sort_values(["total", "clase_real"], ascending=[False, True])
        .head(top)
        .reset_index(drop=True)
    )


def unordered_pairs(confusions: pd.DataFrame, top: int = 4) -> pd.DataFrame:
    """Suma las confusiones en los dos sentidos (A → B y B → A) sobre todos los modelos.

    ``confusions`` es la salida de ``confusion_pairs`` (conviene pedirla con ``top``
    grande). Devuelve los ``top`` pares con más confusiones en total.
    """
    pares = confusions.assign(
        clase_a=confusions[["clase_real", "clase_predicha"]].min(axis=1),
        clase_b=confusions[["clase_real", "clase_predicha"]].max(axis=1),
    )
    return (
        pares.groupby(["clase_a", "clase_b"], as_index=False)["total"]
        .sum()
        .sort_values(["total", "clase_a"], ascending=[False, True])
        .head(top)
        .reset_index(drop=True)
    )


def convergence(histories: dict[str, pd.DataFrame], ultimas: int = 3) -> pd.DataFrame:
    """Si cada corrida había dejado de mejorar cuando terminó.

    ``mejora_ultimas`` es cuánto subió el F1 de validación en las últimas ``ultimas``
    épocas: si sigue siendo positiva al llegar al tope de épocas, el modelo no convergió
    y su resultado es una cota inferior de lo que la arquitectura puede dar.
    """
    filas = {}
    for m, h in histories.items():
        f1 = h["eval_f1_macro"].to_numpy()
        loss = h["eval_loss"].to_numpy()
        filas[m] = {
            "epocas": len(f1),
            "mejor_val_f1": float(f1.max()),
            "epoca_mejor": int(f1.argmax()) + 1,
            "val_f1_final": float(f1[-1]),
            "mejora_ultimas": float(f1[-1] - f1[-1 - ultimas]) if len(f1) > ultimas else None,
            "val_loss_min": float(loss.min()),
            "epoca_val_loss_min": int(loss.argmin()) + 1,
            "val_loss_final": float(loss[-1]),
        }
    return pd.DataFrame(filas).T.infer_objects()


# --------------------------------------------------------------------------------------
# Varias recetas por arquitectura
# --------------------------------------------------------------------------------------


def fmt_lr(lr: float) -> str:
    """``5e-05`` → ``5e-5``: notación científica sin ceros en el exponente."""
    mantisa, exponente = f"{lr:.0e}".split("e")
    return f"{mantisa}e{int(exponente)}"


def run_table(metrics: dict[str, dict]) -> pd.DataFrame:
    """Una fila por corrida: arquitectura, factores de la receta y resultado."""
    filas = {}
    for run, d in metrics.items():
        receta = d["receta"]
        filas[run] = {
            "arquitectura": d["model_key"],
            "checkpoint": d["checkpoint"],
            "learning_rate": receta["learning_rate"],
            "epochs": receta["epochs"],
            "epochs_entrenadas": receta.get("epochs_entrenadas"),
            "mejor_val_f1": d["best_val_f1_macro"],
            "accuracy": d["test"]["accuracy"],
            "f1_macro": d["test"]["f1_macro"],
            "horas_entrenamiento": d["tiempo_entrenamiento_s"] / 3600,
        }
    return pd.DataFrame(filas).T.infer_objects()


def run_label(run: str, metrics: dict[str, dict]) -> str:
    """Nombre legible de una corrida: arquitectura, learning rate y tope de épocas."""
    d = metrics[run]
    receta = d["receta"]
    return f"{model_name(d['model_key'])} - lr {fmt_lr(receta['learning_rate'])} - {receta['epochs']} ép."


def best_runs(metrics: dict[str, dict], by: str = "best_val_f1_macro") -> dict[str, str]:
    """La mejor corrida de cada arquitectura según ``by`` (por defecto, el F1 de validación).

    Elegir por validación y no por test es lo que permite seguir leyendo el test como una
    estimación honesta: el test no participó de la elección.
    """
    mejores: dict[str, str] = {}
    for run, d in metrics.items():
        arq = d["model_key"]
        if arq not in mejores or d[by] > metrics[mejores[arq]][by]:
            mejores[arq] = run
    return mejores


def plot_recipe_curves(histories: dict[str, pd.DataFrame], metrics: dict[str, dict]) -> plt.Figure:
    """F1 de validación por época, un panel por arquitectura y una línea por receta."""
    arquitecturas = list(dict.fromkeys(metrics[r]["model_key"] for r in histories))
    arquitecturas.sort(key=lambda a: (list(MODEL_COLORS).index(a) if a in MODEL_COLORS else 99, a))
    estilos = ("-", "--", ":", "-.")
    fig, axes = plt.subplots(
        1, len(arquitecturas), figsize=(4.2 * len(arquitecturas), 4), sharey=True
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")
    for ax, arq in zip(axes, arquitecturas):
        corridas = [r for r in histories if metrics[r]["model_key"] == arq]
        for i, run in enumerate(corridas):
            h = histories[run]
            receta = metrics[run]["receta"]
            ax.plot(
                h["epoch"],
                h["eval_f1_macro"],
                color=_color(arq),
                linewidth=2,
                linestyle=estilos[i % len(estilos)],
                label=f"lr {fmt_lr(receta['learning_rate'])} - {receta['epochs']} ép.",
            )
            mejor = h["eval_f1_macro"].idxmax()
            ax.scatter(
                h.loc[mejor, "epoch"],
                h.loc[mejor, "eval_f1_macro"],
                s=60,
                color=_color(arq),
                edgecolor="white",
                linewidth=1.5,
                zorder=3,
            )
        _style(ax, model_name(arq), "época", "F1 macro de validación" if ax is axes[0] else "")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    _suptitle(fig, "Efecto de la receta dentro de cada arquitectura")
    fig.tight_layout()
    return fig


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


def _suptitle(fig, text: str) -> None:
    fig.suptitle(text, x=0.02, ha="left", fontsize=12, color=INK, fontweight="bold")


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
    color. Los modelos del frente de Pareto de cada eje llevan un anillo y, si son más
    de uno, una línea punteada que los une (con uno solo no hay nada que unir).
    """
    fig, axes = plt.subplots(1, len(costs), figsize=(4.2 * len(costs), 4), sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")
    for ax, (col, etiqueta) in zip(axes, costs.items()):
        datos = summary[[col, perf_col]].dropna().astype(float)
        frente = pareto_front(datos, col, perf_col)
        pf = datos[frente].sort_values(col)
        ax.plot(pf[col], pf[perf_col], color=INK_MUTED, linewidth=1, linestyle=":", zorder=1)
        ax.scatter(
            pf[col], pf[perf_col], s=260, facecolors="none", edgecolors=INK, linewidth=1, zorder=1
        )
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
                model_name(m),
                (fila[col], fila[perf_col]),
                xytext=(11, -3),
                textcoords="offset points",
                fontsize=9,
                color=INK,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.margins(x=0.2)
        _style(
            ax, etiqueta, f"{etiqueta} (escala log)", "F1 macro (test)" if ax is axes[0] else ""
        )
    anillo = Line2D(
        [],
        [],
        marker="o",
        markersize=12,
        markerfacecolor="none",
        markeredgecolor=INK,
        linestyle=":",
        color=INK_MUTED,
        label="frente de Pareto",
    )
    axes[0].legend(
        handles=[anillo], frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right"
    )
    _suptitle(fig, "Desempeño frente al costo")
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
        ax.scatter(
            xs,
            y,
            s=64,
            color=_color(m),
            edgecolor="white",
            linewidth=2,
            zorder=2,
            label=model_name(m),
        )
        ax.annotate(
            model_name(m),
            (xs[-1], y.iloc[-1]),
            xytext=(8, -3),
            textcoords="offset points",
            fontsize=8.5,
            color=INK,
        )
    ax.set_xticks(x, [TERCIL_NAMES[t] for t in TERCILES])
    ax.set_xlim(-0.5, len(TERCILES) - 0.1)
    _style(
        ax,
        f"Diferencia de accuracy contra {model_name(reference)}, por tercil (IC 95 %)",
        "tercil de dificultad de la clase (EDA)",
        "puntos porcentuales",
    )
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    _suptitle(fig, "Brecha según la dificultad de la clase")
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
            per_class.loc[clase, "label"],
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
        f"Spearman ρ = {res['spearman_rho']:.2f} ({_p_text(res['spearman_p'])})",
        transform=ax.transAxes,
        ha="right",
        fontsize=8.5,
        color=INK,
    )
    _style(
        ax,
        f"F1 de {model_name(model)} menos F1 de {model_name(reference)}, por clase",
        "margen de la clase en el EDA (a la izquierda, más difícil)",
        "puntos de F1",
    )
    _suptitle(fig, f"{model_name(model)}: brecha por clase según su dificultad")
    fig.tight_layout()
    return fig


def plot_validation_curves(histories: dict[str, pd.DataFrame]) -> plt.Figure:
    """F1 macro de validación por época, todos los modelos juntos."""
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
            label=model_name(m),
        )
    # Etiquetas al final de cada curva, separadas para que no se pisen si terminan parejas.
    x_fin = max(h["epoch"].iloc[-1] for h in histories.values())
    finales = sorted(
        ((h["eval_f1_macro"].iloc[-1], m) for m, h in histories.items()), reverse=True
    )
    y_prev = None
    for valor, m in finales:
        y = valor if y_prev is None else min(valor, y_prev - 0.018)
        ax.annotate(
            f"{model_name(m)} {valor:.3f}",
            (x_fin + 0.6, y),
            va="center",
            fontsize=8.5,
            color=INK,
        )
        y_prev = y
    ax.set_xlim(right=x_fin + 6)
    _style(ax, "F1 macro de validación por época", "época", "F1 macro")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    _suptitle(fig, "Convergencia del entrenamiento")
    fig.tight_layout()
    return fig
