"""CLI de verificacion e inspeccion visual de los transforms.

    python -m vit_for_101_food_app.features verify
    python -m vit_for_101_food_app.features preview --model mobilevit --policy standard

``verify`` es el test de equivalencia corriendo como comando, para poder ejecutarlo en
Colab donde no hay pytest configurado.
"""

from pathlib import Path

import typer

from vit_for_101_food_app.config import CACHE_DIR, FIGURES_DIR, MODELS

app = typer.Typer(help="Transforms de preprocesamiento: verificacion y preview.")


@app.command()
def verify(rtol: float = 1e-4, atol: float = 1e-4):
    """Compara la politica 'eval' contra el AutoImageProcessor de cada modelo."""
    import numpy as np
    from PIL import Image
    import torch

    from vit_for_101_food_app.preprocessing import policies, processors

    rng = np.random.default_rng(42)
    imagen = Image.fromarray(rng.integers(0, 256, (400, 600, 3), dtype=np.uint8))

    fallos = 0
    for key in MODELS:
        spec = processors.spec_for(key)
        nuestro = policies.build_transform(spec, "eval")(imagen)
        suyo = processors.load_processor(key)(imagen, return_tensors="pt")["pixel_values"][0]
        try:
            torch.testing.assert_close(nuestro, suyo, rtol=rtol, atol=atol)
            estado = "OK"
        except AssertionError as exc:
            estado, fallos = f"FALLA -- {exc}", fallos + 1
        canales = "BGR" if spec.do_flip_channel_order else "RGB"
        norm = "normaliza" if spec.do_normalize else "NO normaliza"
        typer.echo(f"{key:10s} {spec.target_size}px  {canales}  {norm}  ->  {estado}")

    if fallos:
        raise typer.Exit(code=1)
    typer.echo("todos los transforms coinciden con su AutoImageProcessor")


@app.command()
def preview(
    model: str = "vit",
    policy: str = "standard",
    n: int = 6,
    out_dir: Path = FIGURES_DIR,
):
    """Escribe una grilla antes/despues en reports/figures/."""
    import matplotlib.pyplot as plt
    import torch

    from vit_for_101_food_app.preprocessing import loaders, policies, processors, splits

    spec = processors.spec_for(model)
    frame = splits.load_split("val").head(n)
    _, label2id = splits.load_label_map()
    dataset = loaders.Food101Dataset(
        frame,
        images_root=CACHE_DIR,
        transform=policies.build_transform(spec, policy),
        label2id=label2id,
    )

    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.6))
    for ax, i in zip(axes, range(n)):
        t = dataset[i]["pixel_values"]
        if spec.do_flip_channel_order:
            t = t.flip(-3)
        if spec.do_normalize:
            mean = torch.tensor(spec.image_mean).view(-1, 1, 1)
            std = torch.tensor(spec.image_std).view(-1, 1, 1)
            t = t * std + mean
        ax.imshow(t.permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_title(frame["label"].iloc[i], fontsize=8)
        ax.axis("off")

    fig.suptitle(f"{model} · {policy} · {spec.target_size}px")
    out_dir.mkdir(parents=True, exist_ok=True)
    destino = out_dir / f"preview-{model}-{policy}.png"
    fig.savefig(destino, dpi=120, bbox_inches="tight")
    typer.echo(f"escrito: {destino}")


if __name__ == "__main__":
    app()
