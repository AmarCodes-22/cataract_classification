from typing import Optional
from tqdm import tqdm
from PIL import Image
import numpy as np
import albumentations as A
from operator import xor
import os
from datasets import load_dataset

class ClassificationDatasetPreprocessor:
    def __init__(self, dataset_dir: str, longest_max_size: Optional[int]=None, smallest_max_size: Optional[int]=None, output_dir: Optional[str]=None):
        assert os.path.exists(dataset_dir), f"Dataset directory {dataset_dir} does not exist."
        assert xor(
            longest_max_size is None, smallest_max_size is None,
        ), f"Either longest_max_size or smallest_max_size must be provided, {longest_max_size = }, {smallest_max_size = }"

        self.longest_max_size = longest_max_size
        self.smallest_max_size = smallest_max_size
        self.dataset_dir = dataset_dir
        self.output_dir = os.path.join(os.path.dirname(dataset_dir), os.path.basename(dataset_dir) + '-preprocessed' if output_dir is None else output_dir)

        self.dataset = load_dataset("imagefolder", data_dir=self.dataset_dir, split="train")

        if self.longest_max_size is not None:
            self.transform = A.Compose([A.LongestMaxSize(max_size=self.longest_max_size)])
        elif self.smallest_max_size is not None:
            self.transform = A.Compose([A.SmallestMaxSize(max_size=self.smallest_max_size)])
    

    def process_dataset(self):
        for sample in tqdm(self.dataset):
            input_fpath = sample['image'].filename
            output_fpath = input_fpath.replace(self.dataset_dir, self.output_dir)
            os.makedirs(os.path.dirname(output_fpath), exist_ok=True)

            sample['image'] = Image.fromarray(self.transform(image=np.array(sample['image']))['image'])
            sample['image'].save(output_fpath)
    