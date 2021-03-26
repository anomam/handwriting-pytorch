import math
from abc import ABC, abstractmethod
from typing import NamedTuple, Tuple

import torch
import torch.nn as nn

from handwriting import HandwritingException
from handwriting.config import DIRECTORY_EST
from handwriting.model import Hidden, ModelOutputs

EPSILON = 1e-6


class AbstractEstimator(ABC, nn.Module):
    _n_mdn: int

    def __init__(self, **kwargs):
        super().__init__()
        pass

    @abstractmethod
    def init_hidden(self, batch_size: int, device: torch.device = torch.device("cpu")):
        raise NotImplementedError

    @property
    def n_mdn(self) -> int:
        return self._n_mdn

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMGraves(AbstractEstimator):
    """
    Architecture from paper
    """

    def __init__(self, hidden_size: int = 5, n_mdn: int = 20, num_layers: int = 3):
        super().__init__()
        self._n_mdn = n_mdn
        self._output_size = n_mdn * 6 + 1
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        input_size = 3
        # create LSTM blocks
        lstm_blocks = [nn.LSTM(input_size, hidden_size, 1, batch_first=True)] + [
            nn.LSTM(
                input_size + hidden_size,
                hidden_size,
                1,
                batch_first=True,
            )
            for _ in range(1, num_layers)
        ]
        self.LSTM_blocks = nn.ModuleList(lstm_blocks)
        self.fc_out = nn.Linear(num_layers * hidden_size, self._output_size)

    def forward(
        self, x: torch.Tensor, hidden_layers: Hidden
    ) -> Tuple[torch.Tensor, Hidden]:

        out, hidden = self.LSTM_blocks[0](
            x, (hidden_layers.h[0:1], hidden_layers.c[0:1])
        )
        hidden_out = [out]
        hidden_layers_time = [hidden]
        for i in range(1, self._num_layers):
            inpt = torch.cat((x, out), dim=2)
            out, hidden = self.LSTM_blocks[i](
                inpt, (hidden_layers.h[i : i + 1], hidden_layers.c[i : i + 1])
            )
            hidden_out.append(out)
            hidden_layers_time.append(hidden)
        inpt = torch.cat(hidden_out, dim=2)  # <batch, time, all hidden features concat>
        yhat = self.fc_out(inpt)
        # format hidden
        hs, cs = zip(*hidden_layers_time)
        return yhat, Hidden(torch.cat(hs, dim=0), torch.cat(cs, dim=0))

    def init_hidden(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> Hidden:
        h0 = torch.zeros(
            (self._num_layers, batch_size, self._hidden_size), device=device
        )
        c0 = torch.zeros(
            (self._num_layers, batch_size, self._hidden_size), device=device
        )
        return Hidden(h0, c0)

    @property
    def n_mdn(self) -> int:
        return self._n_mdn

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_modeloutputs_from_yhat(
    yhat: torch.Tensor, n_mdn: int, bias: float = 0.0
) -> ModelOutputs:

    params_split = torch.split(yhat, [n_mdn] * 6 + [1], dim=2)

    return ModelOutputs(
        torch.log_softmax(params_split[0] * (1 + bias), 2),  # log(alpha_i)
        params_split[1],  # mu1
        params_split[2],  # mu2
        params_split[3] - bias,  # log(std1)
        params_split[4] - bias,  # log(std2)
        torch.tanh(params_split[5]),  # rho
        params_split[6],  # phi
    )


def log_gaussian_kernels(targets: torch.Tensor, params: ModelOutputs) -> torch.Tensor:
    """Calculate gaussian kernel for all mdns at the same time

    x: <n features>
    """
    x1, x2 = targets[:, :, 0:1], targets[:, :, 1:2]
    # stds
    sigma_1_is = torch.exp(params.log_sigma_1_is) + EPSILON
    sigma_2_is = torch.exp(params.log_sigma_2_is) + EPSILON
    z = (
        ((x1 - params.mu_1_is) / sigma_1_is).pow(2)
        + ((x2 - params.mu_2_is) / sigma_2_is).pow(2)
        - (2.0 * params.rho_is * (x1 - params.mu_1_is) * (x2 - params.mu_2_is))
        / (sigma_1_is * sigma_2_is)
    )
    x = -z / (2.0 * (1.0 - params.rho_is.pow(2) + EPSILON))
    log_constant = (
        -math.log(2 * math.pi)
        - params.log_sigma_1_is
        - params.log_sigma_2_is
        - 0.5 * torch.log(1 - params.rho_is.pow(2) + EPSILON)
    )
    log_pdfs = log_constant + x
    return log_pdfs


def calculate_log_mdns(targets: torch.Tensor, params: ModelOutputs) -> torch.Tensor:
    log_pdfs = log_gaussian_kernels(targets, params)
    log_weighted_pdfs = params.log_alpha_is + log_pdfs
    return log_weighted_pdfs


def calculate_loss(
    targets: torch.Tensor, params: ModelOutputs, mask: torch.Tensor
) -> torch.Tensor:
    log_weighted_pdfs = calculate_log_mdns(targets, params)
    log_sum_weighted_pdfs = torch.logsumexp(log_weighted_pdfs, 2)
    BCE = nn.BCEWithLogitsLoss(reduction="none")
    lift_pen = targets[:, :, 2:3]
    lift_pen_loss = BCE(params.phi_logit, lift_pen)
    loss = (
        -log_sum_weighted_pdfs + lift_pen_loss[:, :, 0]
    )  # warning: minus sign already included in BCE
    return torch.sum(loss * mask)


def save_estimator(est: AbstractEstimator, filename: str = "estimator"):
    DIRECTORY_EST.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": est.state_dict()}, DIRECTORY_EST / f"{filename}.tar"
    )


def load_estimator(
    lstm_hidden_size: int = 400,
    lstm_num_layers: int = 3,
    n_mixtures: int = 20,
    filename: str = "estimator",
) -> AbstractEstimator:
    """Will raise an error if not found"""
    fp = DIRECTORY_EST / f"{filename}.tar"
    if not fp.exists():
        raise HandwritingException(f"No estimator data found: {fp}")
    est = LSTMGraves(
        hidden_size=lstm_hidden_size, n_mdn=n_mixtures, num_layers=lstm_num_layers
    )
    checkpoint = torch.load(fp)
    est.load_state_dict(checkpoint["model_state_dict"])
    return est
