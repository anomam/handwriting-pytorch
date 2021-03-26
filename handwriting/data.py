import functools
import json
import random
from typing import List

import numpy as np  # type: ignore
import torch
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore

from handwriting import constants, model
from handwriting.config import LOG_PATH


class CustomDataset(Dataset):
    def __init__(
        self,
        array_data: model.ArrayDataNumpy,
        seq_len: int = 300,
        data_augmentation: bool = False,
    ):
        self._array_data = array_data
        self._seq_len = seq_len
        self._data_aug = data_augmentation

    def __len__(self):
        return self._array_data.x.shape[0]  # number of strokesets in dataset

    def __getitem__(self, idx) -> model.ArrayDataNumpy:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ex_len = self._array_data.x_len[idx, 0]
        if self._data_aug and ex_len > self._seq_len:
            start = random.randint(0, ex_len - self._seq_len)
            end = start + self._seq_len
        else:
            start, end = 0, self._seq_len

        return model.ArrayDataNumpy(
            self._array_data.x[idx, start:end, :],
            self._array_data.x_len[idx, :],
            self._array_data.mask[idx, start:end],
        )


def strokesets_to_coords(
    strokesets: List[model.StrokeSet],
) -> model.ArrayDataNumpy:
    max_strokeset_len = constants.MAX_STROKESET_LEN
    # create an array and fill it
    m = len(strokesets)
    x = np.zeros((m, max_strokeset_len, 3), dtype=np.float32)
    x_len = np.zeros((m, 1), dtype=np.int16)
    mask = np.zeros((m, max_strokeset_len), dtype=np.int16)
    for i, strokeset in tqdm(enumerate(strokesets)):
        coords = np.array(
            functools.reduce(
                lambda acc, stroke: acc + stroke, strokeset.strokes, []  # type:ignore
            )
        )
        len_retained = min(max_strokeset_len, coords.shape[0])
        x[i, :len_retained, :] = coords[:len_retained]
        x_len[i, 0] = len_retained
        mask[i, :len_retained] = 1
    return model.ArrayDataNumpy(x, x_len, mask)


def coords_to_offsets(
    array_data: model.ArrayDataNumpy,
) -> model.ArrayDataNumpy:
    """
    coords & offsets dim: <batch, time, features>
    """
    coords = array_data.x
    m = coords.shape[0]
    # offsets are just the diff of coords
    offsets = np.concatenate(
        [coords[:, 1:, :2] - coords[:, :-1, :2], coords[:, 1:, 2:3]], axis=2
    )
    offsets = np.concatenate([np.tile([[0, 0, 1]], (m, 1, 1)), offsets], axis=1)
    x_len = array_data.x_len
    mask_ex_shorter_than_max_time = x_len.squeeze() < offsets.shape[1]
    # the two indices should be of exact same size
    idx_examples = np.where(mask_ex_shorter_than_max_time)[0]  # mask -> indices
    if len(idx_examples):
        idx_time = x_len[mask_ex_shorter_than_max_time, 0]
        offsets[idx_examples, idx_time, :] = 0
    return model.ArrayDataNumpy(offsets, array_data.x_len, array_data.mask)


def offsets_to_coords(array_data: model.ArrayDataNumpy) -> model.ArrayDataNumpy:
    offsets = array_data.x
    coords = np.concatenate(
        [np.cumsum(offsets[:, :, :2], axis=1), offsets[:, :, 2:3]], axis=2
    )
    return model.ArrayDataNumpy(coords, array_data.x_len, array_data.mask)


def offsets_to_batch(
    array_data: model.ArrayDataNumpy, device: torch.device = torch.device("cpu")
) -> model.TrainingBatch:
    if isinstance(array_data.x, np.ndarray):
        # did not come from dataloader
        offsets = torch.tensor(
            array_data.x[:, :-1, :], dtype=torch.float32, device=device
        )
        targets = torch.tensor(
            array_data.x[:, 1:, :], dtype=torch.float32, device=device
        )
        mask = torch.tensor(array_data.mask[:, 1:], device=device)
        x_len = torch.tensor(array_data.x_len)
    else:
        offsets = array_data.x[:, :-1, :].type(torch.float32).to(device)
        targets = array_data.x[:, 1:, :].type(torch.float32).to(device)
        mask = array_data.mask[:, 1:].type(torch.float32).to(device)
        x_len = array_data.x_len

    return model.TrainingBatch(offsets, targets, x_len, mask)


def normalize(
    data: model.ArrayDataNumpy, filename: str = "normalize"
) -> model.ArrayDataNumpy:

    # calculate normal value and save
    reshaped_x = data.x.reshape(-1, 3)
    reshaped_mask = data.mask.reshape(-1).astype(bool)
    mean = np.mean(reshaped_x[reshaped_mask, :], axis=0)
    std = np.std(reshaped_x[reshaped_mask, :], axis=0)
    # need to be applied only on x1, x2, not on pen
    mean[2:] = 0.0
    std[2:] = 1.0
    # write this file into the log folder
    with open(LOG_PATH / (filename + ".json"), "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    # reshape to apply only to selected items
    reshaped_x = data.x.reshape(-1, 3)
    reshaped_mask = data.mask.reshape(-1).astype(bool)
    reshaped_x[reshaped_mask, :] -= mean
    reshaped_x[reshaped_mask, :] /= std
    return model.ArrayDataNumpy(reshaped_x.reshape(data.x.shape), data.x_len, data.mask)


def denormalize(
    data: model.ArrayDataNumpy, filename: str = "normalize"
) -> model.ArrayDataNumpy:
    # Get mean and std
    with open(LOG_PATH / (filename + ".json")) as f:
        params = json.load(f)
    mean = np.array(params["mean"], dtype="float32")
    std = np.array(params["std"], dtype="float32")
    # apply only to the actual sequences, not the padding
    if data.mask.ndim:
        reshaped_x = data.x.reshape(-1, 3)
        reshaped_mask = data.mask.reshape(-1).astype(bool)
        reshaped_x[reshaped_mask, :] *= std
        reshaped_x[reshaped_mask, :] += mean
        x = reshaped_x.reshape(data.x.shape)
    else:
        x = mean + data.x * std
    return model.ArrayDataNumpy(x, *data[1:])


def align_coords(data: model.ArrayDataNumpy) -> model.ArrayDataNumpy:
    x = np.copy(data.x)
    for i, coords in enumerate(x):
        x_len = data.x_len[i, 0]
        x[i, :x_len, :] = align(coords[:x_len, :])
    return model.ArrayDataNumpy(x, data.x_len, data.mask)


def align(coords):
    """
    credit: taken from sjvasquez repo
    """
    coords = np.copy(coords)
    X, Y = coords[:, 0].reshape(-1, 1), coords[:, 1].reshape(-1, 1)
    X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)
    offset, slope = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y).squeeze()
    theta = np.arctan(slope)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    coords[:, :2] = np.dot(coords[:, :2], rotation_matrix) - offset
    return coords
