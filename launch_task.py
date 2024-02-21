import hydra
from omegaconf import DictConfig, OmegaConf

from dojo import (
    export_classification,
    predict_classification,
    test_classification,
    train_classification,
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def launch_task(cfg: DictConfig) -> None:
    print("Launching task with config:")
    print(OmegaConf.to_yaml(cfg), end="\n")

    if cfg.project_type == "classification":
        if cfg.task == "train":
            train_classification(cfg)
        pass
    else:
        pass


if __name__ == "__main__":
    launch_task()
