from collections import namedtuple
from typing import List, NamedTuple, Optional

import numpy as np  # type:ignore
import torch

# --- model for raw data


class Point(NamedTuple):
    """A stroke is made of points"""

    x: int
    y: int
    is_last_stroke_point: int


class StrokeSet(NamedTuple):
    """A strokeset is made of multiple strokes"""

    strokes: List[List[Point]]
    key: Optional[str] = None


# --- model for processed data


class ArrayDataNumpy(NamedTuple):
    x: np.ndarray  # <batch, time, 3>
    x_len: np.ndarray = np.array(float("inf"))  # <batch, 1>
    mask: np.ndarray = np.array(float("inf"))  # <batch, time>


class TrainingBatch(NamedTuple):
    offsets: torch.Tensor  # <batch, time, 3>
    targets: torch.Tensor  # <batch, time, 3>
    time_lengths: torch.Tensor  # <batch, 1>
    mask: torch.Tensor  # <batch size, time>


class ModelOutputs(NamedTuple):
    log_alpha_is: torch.Tensor
    mu_1_is: torch.Tensor
    mu_2_is: torch.Tensor
    log_sigma_1_is: torch.Tensor
    log_sigma_2_is: torch.Tensor
    rho_is: torch.Tensor
    phi_logit: torch.Tensor


class Hidden(NamedTuple):
    h: torch.Tensor = torch.tensor(-1)
    c: torch.Tensor = torch.tensor(-1)
