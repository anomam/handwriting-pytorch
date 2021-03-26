import json
from abc import ABC, abstractclassmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union
from xml.etree import ElementTree as ET

import numpy as np  # type:ignore
import torch
from torch.utils.data import Dataset

from handwriting import model
from handwriting.config import DATA_PATH
from handwriting.constants import FN_MASK_DATA, FN_X_DATA, FN_X_LEN_DATA
from handwriting.data import CustomDataset

TYPE_PATH = Union[str, Path]


class DataRepository:
    def __init__(self):
        self._linestrokes_path = DATA_PATH / "data_raw" / "lineStrokes"
        self._np_path = DATA_PATH / "data_np_offsets"

    def get_all_examples(self) -> List[model.StrokeSet]:
        return [self._read_xml(fp) for fp in self._linestrokes_path.rglob("*.xml")]

    def save_numpy_data(self, array_data: model.ArrayDataNumpy) -> None:
        # create folders if necessary
        folder_path = self._np_path
        folder_path.mkdir(parents=True, exist_ok=True)
        # save the build data
        np.save(folder_path / FN_X_DATA, array_data.x, allow_pickle=False)
        np.save(folder_path / FN_X_LEN_DATA, array_data.x_len, allow_pickle=False)
        np.save(folder_path / FN_MASK_DATA, array_data.mask, allow_pickle=False)

    def get_numpy_data(self) -> model.ArrayDataNumpy:
        folder_path = self._np_path
        x = np.load(folder_path / FN_X_DATA, allow_pickle=False)
        x_len = np.load(folder_path / FN_X_LEN_DATA, allow_pickle=False)
        mask = np.load(folder_path / FN_MASK_DATA, allow_pickle=False)
        return model.ArrayDataNumpy(x, x_len, mask)

    def get_numpy_as_dataset(self, seq_len: int = 300) -> Dataset:
        array_data = self.get_numpy_data()
        return CustomDataset(array_data, seq_len=seq_len)

    @staticmethod
    def _read_xml(filepath: TYPE_PATH) -> model.StrokeSet:
        tree = ET.parse(filepath).getroot()
        strokeset_el = tree.find("StrokeSet")
        assert strokeset_el is not None, "No stroke set found in file {}".format(
            filepath
        )
        strokeset_list = [
            [
                model.Point(
                    int(point_el.attrib["x"]),
                    int(point_el.attrib["y"]),
                    int(point_idx == len(stroke_el) - 1),
                )
                for point_idx, point_el in enumerate(stroke_el)
            ]
            for stroke_el in strokeset_el
        ]
        return model.StrokeSet(strokeset_list, Path(filepath).stem)
