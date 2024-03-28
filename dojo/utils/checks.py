from typing import List

from dojo.callbacks import key_to_callback_class

VALID_VERTICAL_NAMES = ["auto", "ecom", "food", "common"]
VALID_PROJECT_TYPES = ["classification", "detection", "segmentation", "generative"]


def assert_valid_config(cfg):
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
        assert (
            key in key_to_callback_class
        ), f"Invalid callback key: {key}. Must be one of {tuple(key_to_callback_class.keys())}"
