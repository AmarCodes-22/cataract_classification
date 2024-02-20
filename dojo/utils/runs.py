import os
from typing import Literal


def get_exp_dir(trainer, stage: Literal["fit", "test", "predict"], experiments_dir="runs"):
    return os.path.join(
        experiments_dir,
        stage,
        f"{trainer.logger.experiment.name}-{trainer.logger.experiment.id}",
        trainer.ckpt_path.split(os.sep)[-1].split(".")[0].split("_")[-1],
    )
