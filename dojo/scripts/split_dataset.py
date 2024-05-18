import argparse
import os

from datasets import load_dataset
from tqdm import tqdm

from dojo.utils import split_hf_dataset

# todo: make this a script for preprocessing dataset and use flags for resizing, splitting, etc.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for preprocessing dataset")
    parser.add_argument("--input_dir", type=str, help="Path to the input directory")
    parser.add_argument("--split_dir", type=str, help="Path to the split directory")
    parser.add_argument("--test_size", type=float, default=0.05, help="Test size for splitting the dataset")

    args = parser.parse_args()

    dataset = load_dataset("imagefolder", data_dir=args.input_dir, split="train")
    split_dataset = split_hf_dataset(
        dataset, test_size=args.test_size, shuffle=True, seed=42, stratify_by_column="label"
    )

    train_dataset, test_dataset = split_dataset["train"], split_dataset["test"]
    classnames = dataset.features["label"].names

    for split in ["train", "test"]:
        dataset = split_dataset[split]

        for sample in tqdm(dataset, desc=f"Saving {split}"):
            save_fpath = os.path.join(
                args.split_dir, split, classnames[int(sample["label"])], os.path.basename(sample["image"].filename)
            )
            os.makedirs(os.path.dirname(save_fpath), exist_ok=True)

            sample["image"].save(save_fpath)
