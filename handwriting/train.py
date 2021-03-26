from typing import NamedTuple, Tuple

import numpy as np  # type: ignore
import torch
import torch.nn as nn
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from handwriting import model
from handwriting.data import offsets_to_batch
from handwriting.estimator import (
    AbstractEstimator,
    calculate_loss,
    create_modeloutputs_from_yhat,
)


def train_one_batch(
    training_batch: model.TrainingBatch,
    net: AbstractEstimator,
    optimizer: Optimizer,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    # torch init
    net.zero_grad()
    optimizer.zero_grad()
    loss_sum = calculate_batch_loss(training_batch, net, train=True, device=device)
    # back propagate in both time AND batch dimensions
    loss_sum.backward()
    nn.utils.clip_grad_value_(net.parameters(), 10)  # from paper
    optimizer.step()
    return loss_sum


def calculate_batch_loss(
    batch: model.TrainingBatch,
    net: AbstractEstimator,
    train: bool = True,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Returns the sum of losses for batch
    Used for both:
    - training
    - validation
    """
    offsets, targets, _, mask = batch
    batch_size = offsets.size(0)
    hidden = net.init_hidden(batch_size, device=device)
    yhat, hidden = net.forward(offsets, hidden)  # forward in time
    if train:
        yhat.register_hook(lambda grad: torch.clamp(grad, -100, 100))  # from paper
    params = create_modeloutputs_from_yhat(yhat, net.n_mdn)
    loss_sum = calculate_loss(targets, params, mask)
    return loss_sum
