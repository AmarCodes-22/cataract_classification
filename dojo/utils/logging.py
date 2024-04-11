from typing import Optional

import wandb
from lightning.pytorch.loggers import WandbLogger


def initialize_wandb_logger(project: str, id: str, resume: bool):
    print("Initializing Weights & Biases logger", end="\n\n")

    return WandbLogger(project=project, id=id, resume=resume, dir="wandb_logs")


def use_artifact(
    artifact_name: str,
    artifact_type: str,
    artifact_reference: str,
    use_checksum: bool,
    logger: WandbLogger,
    max_objects: int,
):
    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_reference(artifact_reference, checksum=use_checksum, max_objects=max_objects)
    logger.use_artifact(artifact)


def log_artifact(
    artifact_name: str,
    artifact_type: str,
    artifact_reference: str,
    use_checksum: bool,
    logger: WandbLogger,
    metadata_dict: Optional[dict] = None,
):
    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_reference(artifact_reference, checksum=use_checksum)

    if metadata_dict is not None:
        artifact.metadata.update(metadata_dict)

    logger.experiment.log_artifact(artifact)
