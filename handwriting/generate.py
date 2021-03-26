from typing import List

import numpy as np  # type: ignore
import torch
from torch.distributions import bernoulli, categorical, multivariate_normal

from handwriting import data, model
from handwriting.estimator import AbstractEstimator, create_modeloutputs_from_yhat


def generate_coords_sequences(
    est: AbstractEstimator,
    batch_size: int = 1,
    seq_len: int = 100,
    bias: float = 0.0,
    normalize_filename: str = "normalize",
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Can generate multiple sequences in parallel (matrix)"""
    # generate offsets from model
    offsets = generate_offset_sequences(
        est, seq_len=seq_len, batch_size=batch_size, bias=bias, device=device
    )
    data_offsets = model.ArrayDataNumpy(offsets)
    data_offsets = data.denormalize(data_offsets, filename=normalize_filename)
    data_coords = data.offsets_to_coords(data_offsets)
    return data_coords.x


def generate_offset_sequences(
    net: AbstractEstimator,
    seq_len: int = 300,
    batch_size: int = 1,
    bias: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Can generate multiple sequences in parallel (matrix)"""
    net.eval()
    n_mdn = net.n_mdn
    # start should be new pen lift
    offsets = torch.cat(
        [torch.zeros((batch_size, 1, 2)), torch.ones((batch_size, 1, 1))], dim=2
    ).to(device)
    hidden = net.init_hidden(batch_size, device=device)
    sequences: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(seq_len):
            yhat, hidden = net.forward(offsets, hidden)
            params = create_modeloutputs_from_yhat(yhat, n_mdn, bias=bias)
            offsets = sample_gaussian_kernels(params, device=device)
            sequences.append(offsets)
    return torch.cat(sequences, dim=1).detach().cpu().numpy()


def sample_gaussian_kernels(
    params: model.ModelOutputs, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    params should have: <batches, 1 timestep, features>
    """
    batch_size = params.mu_1_is.size(0)
    phis = torch.sigmoid(params.phi_logit[:, 0, :])  # end of stroke
    Bernouilli = bernoulli.Bernoulli(probs=phis)
    eos_samples = Bernouilli.sample().squeeze()  # size = <batch size>
    # multinomial: only 1 gaussian will be selected
    Multinomial = categorical.Categorical(logits=params.log_alpha_is[:, 0, :])
    i_gaussians = Multinomial.sample().squeeze()  # size = <batch size>
    # select the appropriate gaussians for each batch example
    slice_batches = torch.arange(batch_size)
    mu_1_is = params.mu_1_is[slice_batches, 0, i_gaussians]
    mu_2_is = params.mu_2_is[slice_batches, 0, i_gaussians]
    sigma_1_is = torch.exp(params.log_sigma_1_is[slice_batches, 0, i_gaussians])
    sigma_2_is = torch.exp(params.log_sigma_2_is[slice_batches, 0, i_gaussians])
    rho_is = params.rho_is[slice_batches, 0, i_gaussians]
    # create gaussian parameters
    mu_k = mu_1_is.new_zeros((batch_size, 2))
    mu_k[:, 0], mu_k[:, 1] = mu_1_is, mu_2_is
    cov_k = mu_1_is.new_zeros((batch_size, 2, 2))
    cov_k[:, 0, 0], cov_k[:, 1, 1] = sigma_1_is.pow(2), sigma_2_is.pow(2)
    cov_k[:, 1, 0] = cov_k[:, 0, 1] = rho_is * sigma_1_is * sigma_2_is
    # Sample from gaussians starting with standard normal
    Z_normal = torch.normal(
        mean=torch.zeros(batch_size, 2, 1), std=torch.ones(batch_size, 2, 1)
    ).to(mu_k.device)
    x = mu_k + torch.matmul(cov_k, Z_normal).squeeze()
    # return sample
    batch_sample = torch.zeros((batch_size, 1, 3), device=device)
    batch_sample[:, 0, 2] = eos_samples.to(device)
    batch_sample[:, 0, :2] = x.to(device)
    return batch_sample
