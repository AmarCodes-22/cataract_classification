import os
from dataclasses import dataclass
from typing import List, Optional

from lightning import Trainer
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

import wandb
from dojo.callbacks import key_to_callback_class, load_checkpoint_callbacks
from dojo.datasets import (
    ClassificationDatasetPreprocessor,
    ClassificationLitDataModule,
    initialize_classification_lit_datamodule,
)
from dojo.logging import (
    MODEL_RAW_ARTIFACT_NAME,
    MODEL_RAW_ARTIFACT_TYPE,
    get_details_from_model_path,
    initialize_wandb_logger,
    log_artifact,
    use_artifact,
)
from dojo.models import ClassificationLitModule, initialize_classification_lit_module
from dojo.utils import get_exp_dir, get_resume_ckpt_fpath


@dataclass
class Modules:
    logger: WandbLogger
    trainer: Trainer
    callbacks: List[Callback]
    model: Optional[ClassificationLitModule] = None
    dataset: Optional[ClassificationLitDataModule] = None
    resume_ckpt_fpath: Optional[str] = None
    preprocessor: Optional[ClassificationDatasetPreprocessor] = None


def initialize_modules(
    cfg, load_model: bool = True, load_dataset: bool = True, load_preprocessor: bool = False
) -> Modules:
    logger = initialize_wandb_logger(**cfg.logger)

    exp_dir = get_exp_dir(logger)
    resume_ckpt_fpath = get_resume_ckpt_fpath(cfg, exp_dir)

    if resume_ckpt_fpath is not None:
        resume_ckpt_epoch = get_details_from_model_path(resume_ckpt_fpath).epoch
        cfg.trainer.max_epochs += resume_ckpt_epoch
        print(
            f"Updating max_epochs to {cfg.trainer.max_epochs} to account for resumed checkpoint resumed from {resume_ckpt_epoch}."
        )

    if load_model:
        model = initialize_classification_lit_module(resume_ckpt_fpath, **cfg.model)
    else:
        model = None

    if load_dataset:
        dataset = initialize_classification_lit_datamodule(**cfg.dataset)
    else:
        dataset = None

    callbacks = load_checkpoint_callbacks(checkpoints_dir=os.path.join(exp_dir, "fit"), **cfg.callbacks.checkpoints)

    # todo: make each type of callback instantiate and validate themselves based on config values
    for callback in cfg.callbacks.other_callbacks:
        if isinstance(callback, DictConfig):
            callback = OmegaConf.to_container(callback, resolve=True)

            callback_key = str(list(callback.keys())[0])
            callback_kwargs = callback[callback_key]

            callback_class = key_to_callback_class[callback_key]
            callbacks.append(callback_class(**callback_kwargs))
        else:
            callback_class = key_to_callback_class[callback]
            callbacks.append(callback_class())

    trainer = Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    if load_preprocessor:
        preprocessor = ClassificationDatasetPreprocessor(**cfg.preprocess)
    else:
        preprocessor = None

    return Modules(
        logger=logger,
        model=model,
        dataset=dataset,
        trainer=trainer,
        callbacks=callbacks,
        resume_ckpt_fpath=resume_ckpt_fpath,
        preprocessor=preprocessor,
    )


def train(cfg):
    modules = initialize_modules(cfg)
    assert modules.model is not None, "Model must be loaded for training."
    assert modules.dataset is not None, "Dataset must be loaded for training."

    # todo: find a way to move this into callback
    if modules.resume_ckpt_fpath is not None:
        use_artifact(
            artifact_name=MODEL_RAW_ARTIFACT_NAME(modules.logger),
            artifact_type=MODEL_RAW_ARTIFACT_TYPE,
            artifact_path=modules.resume_ckpt_fpath,
            use_checksum=False,
            logger=modules.logger,
            max_objects=1,
        )

    modules.trainer.fit(model=modules.model, datamodule=modules.dataset, ckpt_path=modules.resume_ckpt_fpath)


def test(cfg):
    modules = initialize_modules(cfg)
    assert modules.model is not None, "Model must be loaded for testing."
    assert modules.dataset is not None, "Dataset must be loaded for testing."

    if modules.resume_ckpt_fpath is not None:
        model_details = get_details_from_model_path(modules.resume_ckpt_fpath)
        use_artifact(
            artifact_name=f"{model_details.artifact_name}:{model_details.artifact_version}",
            artifact_type=MODEL_RAW_ARTIFACT_TYPE,
            artifact_path=modules.resume_ckpt_fpath,
            use_checksum=False,
            logger=modules.logger,
            max_objects=1,
        )

    modules.trainer.test(model=modules.model, datamodule=modules.dataset, ckpt_path=modules.resume_ckpt_fpath)


def predict(cfg):
    modules = initialize_modules(cfg)
    assert modules.model is not None, "Model must be loaded for prediction."
    assert modules.dataset is not None, "Dataset must be loaded for prediction."

    modules.trainer.predict(model=modules.model, datamodule=modules.dataset, ckpt_path=modules.resume_ckpt_fpath)


def export(cfg):
    modules = initialize_modules(cfg)
    assert modules.model is not None, "Model must be loaded for exporting."
    assert modules.resume_ckpt_fpath is not None, "Export requires a checkpoint to be resumed from."

    modules.model.to_onnx(modules.logger, modules.resume_ckpt_fpath)


def preprocess(cfg):
    modules = initialize_modules(cfg, load_model=False, load_dataset=False, load_preprocessor=True)
    assert modules.preprocessor is not None, "Preprocessor must be loaded for preprocessing."

    modules.preprocessor.process_dataset()
    metadata_dict = OmegaConf.to_container(cfg.preprocess, resolve=True)

    use_artifact(
        artifact_name="dataset-raw",
        artifact_type="dataset",
        artifact_path=modules.preprocessor.dataset_dir,
        use_checksum=True,
        logger=modules.logger,
        max_objects=len(modules.preprocessor.dataset),
    )

    log_artifact(
        artifact_name="dataset-preprocessed",
        artifact_type="dataset",
        artifact_path=modules.preprocessor.output_dir,
        use_checksum=True,
        logger=modules.logger,
        max_objects=len(modules.preprocessor.dataset),
        metadata_dict=metadata_dict,
    )
