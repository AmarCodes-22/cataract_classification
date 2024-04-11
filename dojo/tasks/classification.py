import os

import torch
import wandb
from lightning.pytorch import Trainer

from dojo.callbacks import (
    GenerateTestReportCallback,
    key_to_callback_class,
    load_checkpoint_callbacks,
)
from dojo.datasets import initialize_classification_lit_datamodule
from dojo.models import initialize_classification_lit_module
from dojo.utils import (
    get_exp_dir,
    get_resume_ckpt_fpath,
    initialize_wandb_logger,
    use_artifact,
)


# todo: move this to utils
def get_resume_ckpt_epoch(resume_ckpt_fpath):
    return torch.load(resume_ckpt_fpath)["epoch"]


def initialize_modules(cfg):
    logger = initialize_wandb_logger(**cfg.logger)

    exp_dir = get_exp_dir(logger)
    resume_ckpt_fpath = get_resume_ckpt_fpath(cfg, exp_dir)

    if resume_ckpt_fpath is not None:
        use_artifact("model-resume", "model", f"file://{resume_ckpt_fpath}", True, logger, max_objects=1)
        resume_ckpt_epoch = get_resume_ckpt_epoch(resume_ckpt_fpath)
        cfg.trainer.max_epochs += resume_ckpt_epoch
        print(
            f"Updating max_epochs to {cfg.trainer.max_epochs} to account for resumed checkpoint resumed from {resume_ckpt_epoch}."
        )

    model = initialize_classification_lit_module(resume_ckpt_fpath, **cfg.model)
    dataset = initialize_classification_lit_datamodule(**cfg.dataset)

    callbacks = load_checkpoint_callbacks(checkpoints_dir=os.path.join(exp_dir, "fit"), **cfg.callbacks.checkpoints)

    for key in cfg.callbacks.other_callbacks:
        callback_class = key_to_callback_class[key]
        callbacks.append(callback_class())
    print("Initilized callbacks:", callbacks, end="\n\n")

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
