import os
from typing import Literal

from lightning.pytorch.loggers import WandbLogger

# todo: move to logging


def get_exp_dir(logger: WandbLogger, experiments_dir="runs"):
    return os.path.join(
        experiments_dir,
        f"{logger.experiment.name}-{logger.experiment.id}",
    )
