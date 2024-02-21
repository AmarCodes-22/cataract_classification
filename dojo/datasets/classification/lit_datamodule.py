import os
from typing import Optional

import lightning.pytorch as pl
from datasets import load_dataset
from torch.utils.data import DataLoader

from .main import ClassificationDataset


class ClassificationLitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset_dir: Optional[str] = None,
        val_ratio: float = 0.1,
        test_dataset_dir: Optional[str] = None,
        predict_dataset_dir: Optional[str] = None,
        cache: bool = False,
        image_size: int = 224,
        batch_size: int = 16,
        num_workers: int = 16,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        print(f"Initialized {self.__class__.__name__} with the following hyperparameters:")
        print(self.hparams, end="\n\n")

    def setup(self, stage: str):
        if stage == "fit":
            train_dir = self.hparams["train_dataset_dir"]
            val_ratio = self.hparams["val_ratio"]
            assert os.path.exists(train_dir), f"{train_dir = }"
            assert 0 < val_ratio < 1, f"{val_ratio = }"

            dataset = load_dataset("imagefolder", data_dir=train_dir, split="train")
            dataset_split = dataset.train_test_split(test_size=val_ratio, shuffle=True, seed=42)
            self.train_dataset = ClassificationDataset(
                hf_dataset=dataset_split["train"], image_size=self.hparams["image_size"], cache=self.hparams["cache"]
            )
            self.val_dataset = ClassificationDataset(
                hf_dataset=dataset_split["test"], image_size=self.hparams["image_size"], cache=self.hparams["cache"]
            )
            self.dataset_idx_to_class = self.train_dataset.idx_to_class

        elif stage == "test":
            test_dir = self.hparams["test_dataset_dir"]
            assert os.path.exists(test_dir), f"{test_dir = }"

            self.test_dataset = ClassificationDataset(
                dataset_dir=test_dir, image_size=self.hparams["image_size"], cache=self.hparams["cache"]
            )
        elif stage == "predict":
            self.predict_dataset = ClassificationDataset(
                dataset_dir=self.hparams["predict_dataset_dir"],
                image_size=self.hparams["image_size"],
                cache=self.hparams["cache"],
                drop_labels=True,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"]
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"]
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"]
        )

    def state_dict(self):
        state = {"dataset_idx_to_class": self.dataset_idx_to_class}
        return state

    def load_state_dict(self, state_dict):
        self.dataset_idx_to_class = state_dict["dataset_idx_to_class"]
