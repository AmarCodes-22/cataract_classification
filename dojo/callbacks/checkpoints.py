import os
from argparse import ArgumentParser

from lightning.pytorch.callbacks import ModelCheckpoint


def load_checkpoint_callbacks(
    checkpoints_dir: str,
    monitor_metric: str,
    save_best: bool = True,
    save_latest: bool = True,
    save_every_n_epochs: int = 0,
):
    callbacks = list()

    if save_best:
        callbacks.append(
            ModelCheckpoint(
                monitor=monitor_metric,
                save_top_k=1,
                mode="max",
                filename="epoch_best",
                dirpath=checkpoints_dir,
                auto_insert_metric_name=False,
            )
        )

    if save_latest:
        callbacks.append(
            ModelCheckpoint(
                monitor="global_step",
                save_top_k=1,
                mode="max",
                filename="epoch_latest",
                dirpath=checkpoints_dir,
                auto_insert_metric_name=False,
            )
        )

    if save_every_n_epochs > 0:
        callbacks.append(
            ModelCheckpoint(
                save_top_k=-1,
                every_n_epochs=every_n_epochs,
                filename="epoch_{epoch}",
                dirpath=checkpoints_dir,
                auto_insert_metric_name=False,
            )
        )

    return callbacks


def get_checkpoint_options(parser: ArgumentParser):
    gp = parser.add_argument_group("Checkpoint callback arguments")
    gp.add_argument("--checkpoints_dir", type=str, help="Directory where checkpoints will be saved")
    gp.add_argument(
        "--monitor_metric",
        type=str,
        default="val/acc",
        help="Best checkpoint will be saved by comparing this metric, needs to be logged from the LightningModule",
    )
    gp.add_argument("--save_best", action="store_true", default=True, help="Save best epoch")
    gp.add_argument("--save_latest", action="store_true", default=True, help="Save latest epoch")
    gp.add_argument("--save_every_n_epochs", type=int, default=0, help="Create a ckpt every n epochs")
    return parser


def get_experiment_dir(logger, checkpoints_dir, stage):
    assert stage in {"fit", "test"}

    exp_dir = os.path.join("runs", stage, f"{logger.experiment.name}-{logger.experiment.id}")

    if checkpoints_dir is not None:
        exp_dir = os.path.join(checkpoints_dir, exp_dir)

    return exp_dir
