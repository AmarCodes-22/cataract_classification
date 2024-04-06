import albumentations as A


def save_transform(transform, fpath, data_format="yaml"):
    A.save(transform, fpath, data_format=data_format)


def load_transform(fpath, data_format="yaml"):
    return A.load(fpath, data_format=data_format)
