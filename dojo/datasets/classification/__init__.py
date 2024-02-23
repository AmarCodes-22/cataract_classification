from .lit_datamodule import ClassificationLitDataModule


def initialize_classification_lit_datamodule(**kwargs):
    return ClassificationLitDataModule(**kwargs)
