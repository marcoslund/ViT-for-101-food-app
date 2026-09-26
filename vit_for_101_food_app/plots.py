"""Curvas de entrenamiento a partir del ``log_history`` del Trainer de HuggingFace.

python -m vit_for_101_food_app.plots --model mobilevit
python -m vit_for_101_food_app.plots --model mobilevit --tensorboard
"""

import json
from pathlib import Path
import re

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import typer

from vit_for_101_food_app.config import FIGURES_DIR, MODELS, MODELS_DIR, REPORTS_DIR

app = typer.Typer()

TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


def find_trainer_state(path: Path) -> Path:
    """Devuelve el ``trainer_state.json`` del checkpoint mas reciente bajo ``path``."""
    path = Path(path)
    if path.is_file():
        return path
    if (path / "trainer_state.json").is_file():
        return path / "trainer_state.json"
    checkpoints = [p for p in path.glob("checkpoint-*") if (p / "trainer_state.json").is_file()]
    if not checkpoints:
        raise FileNotFoundError(f"no hay checkpoint-*/trainer_state.json en {path}")
    ultimo = max(checkpoints, key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
    return ultimo / "trainer_state.json"


def load_log_history(path: Path) -> list[dict]:
    state = json.loads(find_trainer_state(path).read_text(encoding="utf-8"))
    return state["log_history"]


def history_frames(log_history: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa el ``log_history`` en filas de train y de validacion."""
    history = pd.DataFrame(log_history)
    train = history[history["loss"].notna()] if "loss" in history else history.iloc[0:0]
    val = history[history["eval_loss"].notna()] if "eval_loss" in history else history.iloc[0:0]
    train_cols = [c for c in ("step", "epoch", "loss", "learning_rate", "grad_norm") if c in train]
    val_cols = [c for c in history.columns if c in ("step", "epoch") or c.startswith("eval_")]
    return train[train_cols].reset_index(drop=True), val[val_cols].reset_index(drop=True)


def history_table(log_history: list[dict]) -> pd.DataFrame:
    """Historial en formato largo (una fila por log), apto para guardar como CSV."""
    train, val = history_frames(log_history)
    return (
        pd.concat([train.assign(split="train"), val.assign(split="val")], ignore_index=True)
        .sort_values(["step", "split"])
        .reset_index(drop=True)
    )


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel("epoca", color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)


def _new_figure(title: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, color=INK, fontweight="bold")
    return fig, ax


def plot_loss(log_history: list[dict], title: str) -> plt.Figure:
    """Loss de train contra loss de validacion."""
    train, val = history_frames(log_history)
    fig, ax = _new_figure(title)
    ax.plot(train["epoch"], train["loss"], color=TRAIN_COLOR, linewidth=1.5, label="train")
    ax.plot(
        val["epoch"],
        val["eval_loss"],
        color=VAL_COLOR,
        linewidth=2,
        marker="o",
        markersize=4.5,
        label="validacion",
    )
    if not val.empty:
        mejor_loss = val.loc[val["eval_loss"].idxmin()]
        ax.axvline(mejor_loss["epoch"], color=INK_MUTED, linewidth=0.8, linestyle=":")
        ax.annotate(
            f"min val loss {mejor_loss['eval_loss']:.3f}\n(epoca {mejor_loss['epoch']:.0f})",
            xy=(mejor_loss["epoch"], mejor_loss["eval_loss"]),
            xytext=(8, 28),
            textcoords="offset points",
            fontsize=8.5,
            color=INK,
        )
    _style(ax, "Loss", "cross-entropy")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    return fig


def plot_validation(log_history: list[dict], title: str) -> plt.Figure:
    """F1 macro y accuracy de validacion por epoca."""
    _, val = history_frames(log_history)
    fig, ax = _new_figure(title)
    for col, estilo, nombre in (
        ("eval_f1_macro", "-", "F1 macro"),
        ("eval_accuracy", "--", "accuracy"),
    ):
        if col in val:
            ax.plot(
                val["epoch"],
                val[col],
                color=VAL_COLOR,
                linestyle=estilo,
                linewidth=2,
                marker="o" if estilo == "-" else None,
                markersize=4.5,
                label=nombre,
            )
    if "eval_f1_macro" in val and not val.empty:
        mejor = val.loc[val["eval_f1_macro"].idxmax()]
        ax.axvline(mejor["epoch"], color=INK_MUTED, linewidth=0.8, linestyle=":")
        ax.annotate(
            f"mejor F1 {mejor['eval_f1_macro']:.3f}\n(epoca {mejor['epoch']:.0f})",
            xy=(mejor["epoch"], mejor["eval_f1_macro"]),
            xytext=(-78, -38),
            textcoords="offset points",
            fontsize=8.5,
            color=INK,
        )
    _style(ax, "Validacion", "")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    return fig


def plot_learning_rate(log_history: list[dict], title: str) -> plt.Figure:
    """Learning rate por paso (warmup + decaimiento lineal de la receta)."""
    train, _ = history_frames(log_history)
    fig, ax = _new_figure(title)
    if "learning_rate" in train:
        ax.plot(train["epoch"], train["learning_rate"] * 1e4, color=TRAIN_COLOR, linewidth=2)
    _style(ax, "Learning rate", "x 1e-4")
    fig.tight_layout()
    return fig


def plot_training_curves(
    log_history: list[dict], title: str, output_dir: Path | None = None, prefix: str = "curvas"
) -> dict[str, Path]:
    """Genera y guarda las figuras de loss, validacion y learning rate."""
    figuras = {
        "loss": plot_loss(log_history, title),
        "validacion": plot_validation(log_history, title),
        "lr": plot_learning_rate(log_history, title),
    }
    rutas = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for nombre, fig in figuras.items():
            rutas[nombre] = output_dir / f"{prefix}-{nombre}.png"
            fig.savefig(rutas[nombre], dpi=150, bbox_inches="tight")
        logger.success(f"curvas guardadas en {output_dir} ({prefix}-*.png)")
    return rutas


def export_to_tensorboard(log_history: list[dict], logdir: Path) -> Path:
    """Reescribe el ``log_history`` como eventos de TensorBoard."""
    from torch.utils.tensorboard import SummaryWriter

    logdir = Path(logdir)
    writer = SummaryWriter(log_dir=str(logdir))
    for fila in log_history:
        step = fila.get("step")
        if step is None:
            continue
        for clave, valor in fila.items():
            if clave in ("step", "epoch") or not isinstance(valor, (int, float)):
                continue
            tag = f"eval/{clave[5:]}" if clave.startswith("eval_") else f"train/{clave}"
            writer.add_scalar(tag, valor, step)
        if "epoch" in fila:
            writer.add_scalar("train/epoch", fila["epoch"], step)
    writer.close()
    logger.success(f"logs de TensorBoard en {logdir}")
    return logdir


@app.command()
def main(
    model: str = typer.Option(..., "--model", "-m", help=f"clave del registry: {sorted(MODELS)}"),
    run_dir: Path = typer.Option(
        None, help="carpeta con checkpoints (default: models/<model>/full)"
    ),
    tensorboard: bool = typer.Option(False, help="ademas, reconstruir logs de TensorBoard"),
):
    run_dir = run_dir or MODELS_DIR / model / "full"
    log_history = load_log_history(run_dir)

    destino_csv = REPORTS_DIR / "results" / model / "training_history.csv"
    destino_csv.parent.mkdir(parents=True, exist_ok=True)
    history_table(log_history).to_csv(destino_csv, index=False)
    logger.success(f"historial guardado en {destino_csv}")

    plot_training_curves(
        log_history, f"{model} - {MODELS.get(model, '')}", FIGURES_DIR, prefix=f"curvas-{model}"
    )
    if tensorboard:
        export_to_tensorboard(log_history, run_dir / "runs" / "reconstruido")


if __name__ == "__main__":
    app()
