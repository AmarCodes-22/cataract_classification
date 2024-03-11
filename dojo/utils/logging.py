from lightning.pytorch.loggers import WandbLogger

import wandb


def initialize_wandb_logger(project: str, id: str, resume: bool):
    print("Initializing Weights & Biases logger", end="\n\n")

    return WandbLogger(project=project, id=id, resume=resume)


def use_artifact(
    artifact_name: str, artifact_type: str, artifact_reference: str, use_checksum: bool, logger: WandbLogger
):
    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_reference(artifact_reference, checksum=use_checksum)
    logger.use_artifact(artifact)
