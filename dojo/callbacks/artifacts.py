import os

from lightning.pytorch.callbacks import Callback

from dojo.logging import (
    DATASET_PREPROCESSED_ARTIFACT_NAME,
    DATASET_PREPROCESSED_ARTIFACT_TYPE,
    MODEL_RAW_ARTIFACT_NAME,
    MODEL_RAW_ARTIFACT_TYPE,
    epoch_to_dojo_alias_version,
    get_details_from_model_path,
    log_artifact,
    use_artifact,
)


class LogArtifactsCallback(Callback):
    def __init__(self):
        pass

    def _use_dataset_artifact(self, trainer, pl_module, dataset_dir, max_objects):
        use_artifact(
            artifact_name=DATASET_PREPROCESSED_ARTIFACT_NAME,
            artifact_type=DATASET_PREPROCESSED_ARTIFACT_TYPE,
            artifact_path=dataset_dir,
            use_checksum=False,
            logger=trainer.logger,
            max_objects=max_objects,
        )

    def on_fit_start(self, trainer, pl_module):
        train_dataset_dir = trainer.datamodule.hparams["train_dataset_dir"]

        self._use_dataset_artifact(
            trainer,
            pl_module,
            dataset_dir=train_dataset_dir,
            max_objects=len(trainer.datamodule.train_dataset) + len(trainer.datamodule.val_dataset),
        )

    def on_test_start(self, trainer, pl_module):
        test_dataset_dir = trainer.datamodule.hparams["test_dataset_dir"]

        self._use_dataset_artifact(
            trainer,
            pl_module,
            dataset_dir=test_dataset_dir,
            max_objects=len(trainer.datamodule.test_dataset),
        )

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        for v in checkpoint["callbacks"].values():
            # we are not tracking changes in the actual weights, only their path
            if v["best_model_path"] and not os.path.exists(v["best_model_path"]):
                epoch = get_details_from_model_path(v["best_model_path"], from_path=True).epoch
                epoch = epoch_to_dojo_alias_version(epoch)
                logger = trainer.logger

                log_artifact(
                    artifact_name=MODEL_RAW_ARTIFACT_NAME(logger),
                    artifact_type=MODEL_RAW_ARTIFACT_TYPE,
                    artifact_path=v["best_model_path"],
                    use_checksum=False,
                    logger=trainer.logger,
                    max_objects=1,
                    metadata_dict={"monitor": v["monitor"], "epoch": epoch},
                    artifact_aliases=[str(epoch)],
                )
