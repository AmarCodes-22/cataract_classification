import os
from operator import xor
from typing import Callable, Optional, Union

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from dojo.utils import load_transform


class ClassificationDataset(TorchDataset):
    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        hf_dataset: Optional[DatasetDict] = None,
        image_size: int = 224,
        cache: bool = False,
        transform: Union[Callable, str] | None = None,
        drop_labels: bool = False,
    ) -> None:
        super().__init__()
        assert xor(
            dataset_dir is None, hf_dataset is None
        ), f"Either dataset_dir or hf_dataset must be provided, {dataset_dir = }, {hf_dataset = }"

        if dataset_dir is not None:
            assert os.path.exists(dataset_dir)
            self.dataset = load_dataset("imagefolder", data_dir=dataset_dir, split="train", drop_labels=drop_labels)
        else:
            self.dataset = hf_dataset

        if not drop_labels:
            self.idx_to_class = {i: name for i, name in enumerate(self.dataset.features["label"].names)}

        util_transform = A.Compose([A.Resize(height=image_size, width=image_size), A.ToFloat(), ToTensorV2()])

        if isinstance(transform, str):
            assert os.path.exists(transform), f"{transform = }"
            transform = load_transform(transform)

        self.transform = transform if transform is not None else util_transform

        self.cache = cache
        self.samples: Union[DatasetDict, dict] = dict()

        if self.cache:
            for sample in tqdm(self.dataset, desc="Caching samples"):
                self.samples["fpath"] = sample["image"]
                self.samples["image"] = sample["image"]
                self.samples["label"] = sample["label"]
        else:
            self.samples = self.dataset

    def __getitem__(self, index):
        result = {}
        sample = self.samples[index]

        result["fpath"] = sample["image"].filename
        image = np.array(sample["image"].convert("RGB"))
        try:
            result["label"] = sample["label"]
        except KeyError:  # drop_labels == True
            pass

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        result["image"] = image
        return result

    def __len__(self):
        return len(self.dataset)
