import hydra
from omegaconf import DictConfig, OmegaConf

from dojo import (
    assert_valid_config,
    export_classification,
    key_to_callback_class,
    predict_classification,
    test_classification,
    train_classification,
)


# todo: assert project names follow a certain convention
@hydra.main(version_base=None, config_path="conf", config_name="config")
def launch_task(cfg: DictConfig) -> None:
    print("Launching task with config:")
    print(OmegaConf.to_yaml(cfg), end="\n")

    assert_valid_config(cfg, list(key_to_callback_class.keys()))

    if cfg.project_type == "classification":
        if cfg.task == "train":
            train_classification(cfg)
        elif cfg.task == "test":
            test_classification(cfg)
        elif cfg.task == "predict":
            predict_classification(cfg)
        elif cfg.task == "export":
            export_classification(cfg)
        pass
    else:
        pass


if __name__ == "__main__":
    launch_task()
