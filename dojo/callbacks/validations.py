from lightning.pytorch.callbacks import Callback


class ValidateArchitectureCallback(Callback):
    def __init__(self):
        pass

    def setup(self, trainer, pl_module, stage):
        num_classes_in_model = pl_module.model.num_classes
        num_classes_in_dataset = trainer.datamodule.num_classes
        assert num_classes_in_model == num_classes_in_dataset, f"{num_classes_in_model = }, {num_classes_in_dataset = }"
