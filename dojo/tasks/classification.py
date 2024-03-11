import os

from lightning.pytorch import Trainer

import wandb
from dojo.callbacks import load_checkpoint_callbacks
from dojo.datasets import initialize_classification_lit_datamodule
from dojo.models import initialize_classification_lit_module
from dojo.utils import get_exp_dir, initialize_wandb_logger


def initialize_modules(cfg):
    logger = initialize_wandb_logger(**cfg.logger)

    exp_dir = get_exp_dir(logger)
    if cfg.resume or cfg.resume_ckpt_fpath is not None:
        if cfg.resume_ckpt_fpath is not None:
            resume_ckpt_fpath = str(cfg.resume_ckpt_fpath)
        else:
            resume_ckpt_fpath = os.path.join(exp_dir, "fit", f"epoch_{cfg.resume_epoch}.ckpt") if cfg.resume else None
        resume_ckpt_fpath = os.path.abspath(resume_ckpt_fpath)
    else:
        resume_ckpt_fpath = None

    if resume_ckpt_fpath is not None:
        artifact = wandb.Artifact(f"model-resume", type="model")

        artifact.add_reference(f"file://{resume_ckpt_fpath}", checksum=True)

        logger.use_artifact(artifact)

    model = initialize_classification_lit_module(resume_ckpt_fpath, **cfg.model)
    dataset = initialize_classification_lit_datamodule(**cfg.dataset)

    callbacks = load_checkpoint_callbacks(checkpoints_dir=os.path.join(exp_dir, "fit"), **cfg.callbacks.checkpoints)

    trainer = Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    return model, dataset, logger, callbacks, trainer, resume_ckpt_fpath


def train(cfg):
    model, dataset, logger, callbacks, trainer, resume_ckpt_fpath = initialize_modules(cfg)
    trainer.fit(model=model, datamodule=dataset, ckpt_path=resume_ckpt_fpath)


def test(cfg):
    model, dataset, logger, callbacks, trainer, resume_ckpt_fpath = initialize_modules(cfg)
    trainer.test(model=model, datamodule=dataset, ckpt_path=resume_ckpt_fpath)


def predict(cfg):
    model, dataset, logger, callbacks, trainer, resume_ckpt_fpath = initialize_modules(cfg)
    trainer.predict(model=model, datamodule=dataset, ckpt_path=resume_ckpt_fpath)


def export(cfg):
    model, dataset, logger, callbacks, trainer, resume_ckpt_fpath = initialize_modules(cfg)
    model.to_onnx(logger, resume_ckpt_fpath)
