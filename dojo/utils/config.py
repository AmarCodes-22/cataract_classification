import os
from typing import List

from omegaconf import DictConfig


def get_resume_ckpt_fpath(cfg, exp_dir):
    if cfg.resume or cfg.resume_ckpt_fpath is not None:
        if cfg.resume_ckpt_fpath is not None:
            resume_ckpt_fpath = str(cfg.resume_ckpt_fpath)
        else:
            resume_ckpt_fpath = os.path.join(exp_dir, "fit", f"epoch_{cfg.resume_epoch}.ckpt") if cfg.resume else None
        resume_ckpt_fpath = os.path.abspath(resume_ckpt_fpath)
    else:
        resume_ckpt_fpath = None

    return resume_ckpt_fpath


VALID_VERTICAL_NAMES = ["auto", "ecom", "food", "common"]
VALID_PROJECT_TYPES = ["classification", "detection", "segmentation", "generative"]


def assert_valid_config(cfg, valid_callback_keys: List[str]):
    # * assert resume options are valid
    if cfg.resume:
        assert cfg.resume_epoch is not None, "If resuming, you must provide the epoch to resume from"

    # * assert project names follow a certain convention
    assert cfg.logger.project.replace("-", "").islower(), "Project names must be lowercase and contain hyphens only"
    assert (
        len(cfg.logger.project.split("-")) > 2
    ), "Project should follow the format: vertical-project_type-name, got {project} instead."

    vertical_name, project_type = cfg.logger.project.split("-")[:2]
    assert (
        vertical_name in VALID_VERTICAL_NAMES
    ), f"Invalid vertical name: {vertical_name}. Must be one of {VALID_VERTICAL_NAMES}"
    assert (
        project_type in VALID_PROJECT_TYPES
    ), f"Invalid project type: {project_type}. Must be one of {VALID_PROJECT_TYPES}"

    for key in cfg.callbacks.other_callbacks:
        if isinstance(key, str):
            assert key in valid_callback_keys, f"Invalid callback key: {key}. Must be one of {valid_callback_keys}"
        elif isinstance(key, DictConfig):
            assert len(key) == 1, f"Invalid callback key: {key}. Must be a dictionary with a single key"
            assert (
                list(key.keys())[0] in valid_callback_keys
            ), f"Invalid callback key: {key}. Must be one of {valid_callback_keys}"
