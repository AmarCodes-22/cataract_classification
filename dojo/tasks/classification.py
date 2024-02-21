import os

from lightning.pytorch import Trainer

from dojo.callbacks import load_checkpoint_callbacks
from dojo.datasets import initialize_classification_lit_datamodule
from dojo.models import initialize_classification_lit_module
from dojo.utils import get_exp_dir, initialize_wandb_logger


def initialize_modules(stage, cfg):
    if cfg.resume:
        assert cfg.resume_epoch is not None, "If resuming, you must provide the epoch to resume from"

    logger = initialize_wandb_logger(**cfg.logger)

    exp_dir = get_exp_dir(logger, stage)
    resume_ckpt_fpath = os.path.join(exp_dir, f"epoch_{cfg.resume_epoch}.ckpt") if cfg.resume else None

    model = initialize_classification_lit_module(resume_ckpt_fpath, **cfg.model)
    dataset = initialize_classification_lit_datamodule(**cfg.dataset)

    callbacks = load_checkpoint_callbacks(checkpoints_dir=exp_dir, **cfg.callbacks.checkpoints)

    trainer = Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    return model, dataset, logger, callbacks, trainer, resume_ckpt_fpath


def train(cfg):
    model, dataset, logger, callbacks, trainer, resume_ckpt_fpath = initialize_modules("fit", cfg)
    trainer.fit(model=model, datamodule=dataset, ckpt_path=resume_ckpt_fpath)


def test():
    pass


def predict():
    pass


def export():
    pass
