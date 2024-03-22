from dojo.utils import split_hf_dataset
from tqdm import tqdm
import os
from datasets import load_dataset

# todo: make this a script for preprocessing dataset and use flags for resizing, splitting, etc.

if __name__ == '__main__':
    input_dir = '/home/ubuntu/members/amar/projects/auto-classification-interior_subroi/data/v1-small'
    split_dir = '/home/ubuntu/members/amar/projects/auto-classification-interior_subroi/data/v1-small-split'
    test_size = 0.1

    dataset = load_dataset("imagefolder", data_dir=input_dir, split="train")
    split_dataset = split_hf_dataset(dataset, test_size=test_size, shuffle=True, seed=42, stratify_by_column="label")

    train_dataset, test_dataset = split_dataset["train"], split_dataset["test"]
    classnames = dataset.features["label"].names

    for split in ['train', 'test']:
        dataset = split_dataset[split]

        for sample in tqdm(dataset, desc=f"Saving {split}"):
            save_fpath = os.path.join(split_dir, split, classnames[int(sample['label'])], os.path.basename(sample['image'].filename))
            os.makedirs(os.path.dirname(save_fpath), exist_ok=True)

            sample['image'].save(save_fpath)
