import os
import shutil
from operator import xor
from typing import Optional

import albumentations as A
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


class ClassificationDatasetPreprocessor:
    def __init__(
        self,
        dataset_dir: str,
        longest_max_size: Optional[int] = None,
        smallest_max_size: Optional[int] = None,
        output_dir: Optional[str] = None,
    ):
        assert os.path.exists(dataset_dir), f"Dataset directory {dataset_dir} does not exist."
        assert xor(
            longest_max_size is None,
            smallest_max_size is None,
        ), f"Either longest_max_size or smallest_max_size must be provided, {longest_max_size = }, {smallest_max_size = }"

        self.longest_max_size = longest_max_size
        self.smallest_max_size = smallest_max_size
        self.dataset_dir = dataset_dir
        self.output_dir = os.path.join(
            os.path.dirname(dataset_dir),
            os.path.basename(dataset_dir) + "-preprocessed" if output_dir is None else output_dir,
        )

        self.dataset = load_dataset("imagefolder", data_dir=self.dataset_dir, split="train")

        if self.longest_max_size is not None:
            self.transform = A.Compose([A.LongestMaxSize(max_size=self.longest_max_size)])
        elif self.smallest_max_size is not None:
            self.transform = A.Compose([A.SmallestMaxSize(max_size=self.smallest_max_size)])

        self.valid_image_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        self.invalid_files_dir = os.path.join(os.path.dirname(dataset_dir), os.path.basename(dataset_dir) + "-invalid")

    def move_non_image_files(self):
        for root, _, fnames in os.walk(self.dataset_dir):
            for fname in tqdm(fnames, desc=f"Moving non-image files"):
                if not fname.endswith(self.valid_image_extensions):
                    fpath = os.path.join(root, fname)
                    out_fpath = fpath.replace(self.dataset_dir, self.invalid_files_dir)
                    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
                    shutil.move(fpath, out_fpath)
                    print(f"Moved {fpath} to {out_fpath} because it is not an image file.")

    def process_dataset(self):
        self.move_non_image_files()

        for sample in tqdm(self.dataset):
            try:
                input_fpath = sample["image"].filename
                output_fpath = self._get_output_fpath(input_fpath)
                os.makedirs(os.path.dirname(output_fpath), exist_ok=True)

                if self._should_skip_sample(sample):
                    continue

                sample["image"] = self._transform_sample_image(sample)
                sample["image"].save(output_fpath)
            except Exception as e:
                print(f"{e = }, {sample = }")

    def _get_output_fpath(self, input_fpath: str):
        return input_fpath.replace(self.dataset_dir, self.output_dir)

    def _should_skip_sample(self, sample):
        if self.smallest_max_size is not None and min(sample["image"].size) < self.smallest_max_size:
            return True
        if self.longest_max_size is not None and max(sample["image"].size) < self.longest_max_size:
            return True
        return False

    def _transform_sample_image(self, sample):
        return Image.fromarray(self.transform(image=np.array(sample["image"]))["image"])
