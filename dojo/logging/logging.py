import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from lightning.pytorch.loggers import WandbLogger

import wandb

DATASET_RAW_ARTIFACT_NAME = "dataset-raw"
DATASET_PREPROCESSED_ARTIFACT_NAME = "dataset-preprocessed"
MODEL_RAW_ARTIFACT_NAME = lambda logger: f"{logger.experiment.name}-{logger.experiment.id}"
MODEL_EXPORT_ARTIFACT_NAME = lambda logger: f"{logger.experiment.name}-{logger.experiment.id}"

DATASET_RAW_ARTIFACT_TYPE = "dataset"
DATASET_PREPROCESSED_ARTIFACT_TYPE = "dataset"
MODEL_RAW_ARTIFACT_TYPE = "model-raw"
MODEL_EXPORT_ARTIFACT_TYPE = "model-export"

epoch_to_dojo_alias_version = lambda epoch: f"dojo_{epoch}"


def initialize_wandb_logger(project: str, id: str, resume: bool, job_type: str):
    print("Initializing Weights & Biases logger", end="\n\n")

    return WandbLogger(project=project, id=id, resume=resume, job_type=job_type)


def use_artifact(
    artifact_name: str,
    artifact_type: str,
    artifact_path: str,
    use_checksum: bool,
    logger: WandbLogger,
    max_objects: int,
):
    try:
        hostname = os.uname()[1]

        if len(artifact_name.split(":")) == 1:  # artifact_name does not contain a version
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_reference(f"dojo://{hostname}:{os.path.abspath(artifact_path)}", name="/", checksum=False)
            logger.use_artifact(artifact)
        else:
            logger.use_artifact(artifact_name)
    except Exception as e:
        print(f"Exception: {e}")


def log_artifact(
    artifact_name: str,
    artifact_type: str,
    artifact_path: str,
    use_checksum: bool,
    logger: WandbLogger,
    max_objects: int,
    metadata_dict: Optional[dict] = None,
    artifact_aliases: Optional[List[str]] = None,
):
    try:
        hostname = os.uname()[1]
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_reference(f"dojo://{hostname}:{os.path.abspath(artifact_path)}", name="/", checksum=False)

        if metadata_dict is not None:
            artifact.metadata.update(metadata_dict)

        logger.experiment.log_artifact(artifact, aliases=artifact_aliases)
    except Exception as e:
        print(f"Exception: {e}")


def get_resume_ckpt_epoch(resume_ckpt_fpath, from_path=False):
    if from_path:
        epoch = os.path.basename(resume_ckpt_fpath).rsplit(".", maxsplit=1)[0].split("_")[-1]
    else:
        epoch = torch.load(resume_ckpt_fpath)["epoch"]

    return epoch


@dataclass
class ModelDetails:
    epoch: int
    artifact_version: str
    artifact_name: str


def get_details_from_model_path(fpath, from_path=False) -> ModelDetails:
    if from_path:
        epoch = os.path.basename(fpath).rsplit(".", maxsplit=1)[0].split("_")[-1]
    else:
        epoch = torch.load(fpath)["epoch"]

    artifact_name = fpath.split(os.sep)[-3]
    artifact_version = epoch_to_dojo_alias_version(epoch)
    return ModelDetails(epoch=epoch, artifact_version=artifact_version, artifact_name=artifact_name)
