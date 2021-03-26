from typing import Optional, Tuple

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import torch

from handwriting import model
from handwriting.config import LOG_PATH
from handwriting.estimator import AbstractEstimator
from handwriting.generate import generate_coords_sequences


def generate_plots(
    est: AbstractEstimator,
    sequence_length: int = 500,
    n_sequences: int = 1,
    bias: float = 0.0,
    rainbow: bool = True,
    linewidth: int = 3,
    normalize_filename: str = "normalize",
    device_str: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    base_plot_name: str = "plot",
):
    device = torch.device(device_str)
    # generate sequences
    coords_sequences = generate_coords_sequences(
        est,
        batch_size=n_sequences,
        seq_len=sequence_length,
        bias=bias,
        normalize_filename=normalize_filename,
        device=device,
    )
    # plot sequences and save them
    fig = plt.figure(figsize=(20, 10))
    dir_plot = LOG_PATH / "plots"
    dir_plot.mkdir(parents=True, exist_ok=True)
    for i, sequence in enumerate(coords_sequences):
        fig.clf()
        ax = fig.add_subplot(111)
        plt.tight_layout()
        make_scaled_coords_plot(sequence, ax, linewidth=linewidth, rainbow=rainbow)
        fig.savefig(
            dir_plot / f"{base_plot_name}_{i}.png", bbox_inches="tight", pad_inches=0.1
        )


def make_scaled_coords_plot(
    coords: np.ndarray,
    ax: plt.Axes,
    scale: float = 1.0,
    rainbow: bool = False,
    linewidth: int = 3,
) -> None:
    """
    coords: <time, features>
    """
    # prepare data
    coords[:, :2] /= scale
    # plot strokes
    stroke = []
    for x, y, eos in coords:
        stroke.append((x, -y))  # need to flip the direction of the y axis
        if eos == 1:
            x, y = zip(*stroke)
            color = np.random.rand(3).tolist() if rainbow else "k"
            ax.plot(x, y, color=color, linewidth=linewidth)
            stroke = []
    if stroke:
        x, y = zip(*stroke)
        color = np.random.rand(3).tolist() if rainbow else "k"
        ax.plot(x, y, color=color, linewidth=linewidth)
        stroke = []
    ax.set_aspect("equal")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
