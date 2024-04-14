# todo: move to datasets


def split_hf_dataset(hf_dataset, **kwargs):
    return hf_dataset.train_test_split(**kwargs)
