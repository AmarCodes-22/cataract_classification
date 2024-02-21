import os
from typing import Literal

from lightning.pytorch.loggers import WandbLogger


def get_exp_dir(logger: WandbLogger, stage: Literal["fit", "test", "predict"], experiments_dir="runs"):
    return os.path.join(
        experiments_dir,
        stage,
        f"{logger.experiment.name}-{logger.experiment.id}",
    )
