from .lit_datamodule import ClassificationLitDataModule
from .preprocess import ClassificationDatasetPreprocessor


def initialize_classification_lit_datamodule(**kwargs):
    return ClassificationLitDataModule(**kwargs)
