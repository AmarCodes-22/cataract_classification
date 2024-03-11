import os
import shutil
import subprocess
import tempfile
from typing import Optional

import lightning as L
import numpy as np
import onnxruntime
import torch
from torcheval.metrics import Mean, MulticlassAccuracy

import wandb
from dojo.utils import get_exp_dir

from .networks import ClassificationModel, ExportWrapper


class ClassificationLitModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        pretrained_model_name_or_path: str = "openai/clip-vit-base-patch32",
        lr: float = 1e-5,
        s3_folder: str = "s3://ai-data-log/dojo-testing",
    ):
        super().__init__()
        lr = float(lr)

        self.model = ClassificationModel(
            pretrained_model_name_or_path=pretrained_model_name_or_path, num_classes=num_classes
        )

        self.loss_fn = self.configure_losses()

        self.example_input_array = torch.randn((4, 3, 224, 224))

        self.save_hyperparameters()

        print(f"Initialized {self.__class__.__name__} with the following hyperparameters:")
        print(self.hparams, end="\n\n")

    def setup(self, stage):
        if stage == "fit":
            self.train_accuracy = MulticlassAccuracy()
            self.validation_accuracy = MulticlassAccuracy()

            self.train_loss = Mean()
            self.validation_loss = Mean()

            self.train_loss.to(self.device)
            self.validation_loss.to(self.device)
        elif stage == "test":
            self.test_accuracy = MulticlassAccuracy()
            self.test_loss = Mean()

            self.test_loss.to(self.device)

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self.forward(images)

        loss = self.loss_fn(logits, labels)

        self.train_loss.update(loss)
        self.train_accuracy.update(self._softmax_argmax(logits), labels)

        self.log("train/acc", self.train_accuracy.compute(), on_step=False, on_epoch=True)
        self.log("train/loss", self.train_loss.compute(), on_step=False, on_epoch=True)
        self.log("global_step", self.trainer.global_step)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self.forward(images)

        loss = self.loss_fn(logits, labels)

        self.validation_loss.update(loss)
        self.validation_accuracy.update(self._softmax_argmax(logits), labels)

        self.log("val/acc", self.validation_accuracy.compute(), on_step=False, on_epoch=True)
        self.log("val/loss", self.validation_loss.compute(), on_step=False, on_epoch=True)
        self.log("global_step", self.trainer.global_step)

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self.forward(images)

        loss = self.loss_fn(logits, labels)

        self.test_loss.update(loss)
        self.test_accuracy.update(self._softmax_argmax(logits), labels)

        self.log("test/acc", self.test_accuracy.compute(), on_step=False, on_epoch=True)
        self.log("test/loss", self.test_loss.compute(), on_step=False, on_epoch=True)
        self.log("global_step", self.trainer.global_step)

        return {"loss": loss}

    # todo: make this a callback
    def on_predict_start(self):
        self.prediction_output_dir = os.path.join(get_exp_dir(self.trainer.logger), "predict")

        self.dataset_idx_to_class = self.trainer.datamodule.dataset_idx_to_class
        for _, v in self.dataset_idx_to_class.items():
            os.makedirs(os.path.join(self.prediction_output_dir, v), exist_ok=True)

    def predict_step(self, batch, batch_idx):
        images = batch["image"]
        logits = self.forward(images)

        preds = self._softmax_argmax(logits)
        return {"preds": preds}

    def on_predict_batch_end(self, output, batch, batch_idx):
        for pred, src_fpath in zip(output["preds"], batch["fpath"]):
            dst_fpath = os.path.join(
                self.prediction_output_dir, self.dataset_idx_to_class[pred.item()], os.path.basename(src_fpath)
            )
            shutil.copy(src_fpath, dst_fpath)

    def on_train_epoch_end(self):
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        self.validation_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        return optimizer

    def configure_losses(self):
        return torch.nn.CrossEntropyLoss()

    def _softmax_argmax(self, logits):
        return torch.argmax(torch.softmax(logits, dim=1), dim=1)

    def to_torchscript(self, resume_ckpt_fpath: str):
        traced_save_fpath = resume_ckpt_fpath.replace(".ckpt", ".pt")

        export_wrapper = ExportWrapper(self.model)
        example_inputs = torch.rand(32, 224, 224, 3, dtype=torch.float32, device=self.device)
        example_inputs = (example_inputs * 255).to(torch.uint8)
        export_wrapper.eval()

        untraced_outputs = export_wrapper(example_inputs)
        print("Output before trace", untraced_outputs.shape, untraced_outputs.dtype)
        print(f"{untraced_outputs = }")

        print("Model input", example_inputs.shape, example_inputs.dtype)
        traced_module = torch.jit.trace(export_wrapper, example_inputs)
        traced_module = torch.jit.optimize_for_inference(traced_module)

        traced_outputs = traced_module(example_inputs)
        print("Output after trace", traced_outputs.shape, traced_outputs.dtype)
        print(f"{traced_outputs = }")

        if traced_save_fpath is not None:
            traced_module.save(traced_save_fpath)

        return traced_module

    def to_onnx(self, logger, resume_ckpt_fpath: str):
        traced_save_fpath = resume_ckpt_fpath.replace(".ckpt", ".onnx")

        export_wrapper = ExportWrapper(self.model)

        example_inputs = torch.rand(32, 224, 224, 3, dtype=torch.float32, device=self.device)
        example_inputs = (example_inputs * 255).to(torch.uint8)
        export_wrapper.eval()

        untraced_output = export_wrapper(example_inputs)
        print("Output before trace", untraced_output.shape, untraced_output.dtype)
        print(f"{untraced_output = }")

        torch.onnx.export(export_wrapper, example_inputs, traced_save_fpath, export_params=True)

        model = onnxruntime.InferenceSession(traced_save_fpath, providers=["CUDAExecutionProvider"])

        input_name = model.get_inputs()[0].name
        traced_output = model.run(None, {input_name: example_inputs.to("cpu").numpy()})

        print("Output after trace", traced_output[0].shape, traced_output[0].dtype)
        print(f"{traced_output = }")

        np.testing.assert_allclose(
            untraced_output.cpu().detach().numpy(),
            traced_output[0],
            rtol=1e-03,
            atol=1e-03,
            err_msg="The outputs do not match!",
        )

        self.log_version(logger, resume_ckpt_fpath)

    def log_version(self, logger: L.pytorch.loggers.WandbLogger, ckpt_fpath: str):
        artifact_type = "model"

        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(self.model.state_dict(), tmp.name)

            # push to s3
            s3_uri = f"{self.hparams['s3_folder']}/{logger.experiment.project}/{artifact_type}/{logger.experiment.name}-{logger.experiment.id}/{os.path.basename(ckpt_fpath)}"
            aws_cli_command = f"aws s3 cp {tmp.name} {s3_uri}"
            subprocess.run(aws_cli_command, shell=True, capture_output=False, text=True)

        # log to wandb as reference artifact
        # todo: add metadata to artifact
        artifact = wandb.Artifact(f"exported-model", type=artifact_type)

        # todo: using checksum=True takes too much time. can i use s3 bucket's metadata instead? (e.g. last modified date, size, etc.)
        artifact.add_reference(s3_uri, checksum=False)

        logger.experiment.log_artifact(artifact)
