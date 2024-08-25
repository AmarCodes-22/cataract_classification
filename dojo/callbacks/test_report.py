import os
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torcheval.metrics import BinaryAUROC, Mean, MulticlassAccuracy
from torcheval.metrics.aggregation.auc import AUC
from tqdm import tqdm

import wandb
from dojo.utils import get_exp_dir, image_tensor_to_pil


class GenerateTestReportCallback(Callback):
    def __init__(self, max_num_images_per_label_pred: int = 20):
        self.test_preds = list()
        self.test_labels = list()
        self.label_pred_to_image = dict()
        self.max_num_images_per_label_pred = max_num_images_per_label_pred

        self.test_auroc = BinaryAUROC()
        self.test_auc = AUC()

    def on_test_batch_end(self, trainer, pl_module, output, batch, batch_idx, dataloader_idx=0):
        self.test_preds.extend(output["preds"].cpu().numpy())
        self.test_labels.extend(output["labels"].cpu().numpy())
        preds, labels = output["preds"].cpu().numpy(), output["labels"].cpu().numpy()
        mismatch_indices = np.where(np.not_equal(preds, labels))[0]
        for index in mismatch_indices:
            image_tensor = batch["image"][index].cpu()
            label = labels[index]
            prediction = preds[index]
            if (label, prediction) not in self.label_pred_to_image:
                self.label_pred_to_image[(label, prediction)] = []
            if len(self.label_pred_to_image[(label, prediction)]) < self.max_num_images_per_label_pred:
                self.label_pred_to_image[(label, prediction)].append(image_tensor)

        self.test_preds_tensor = torch.tensor(self.test_preds)
        self.test_labels_tensor = torch.tensor(self.test_labels)

    def on_test_epoch_end(self, trainer, pl_module):
        # Log metrics
        self.test_auroc.update(self.test_preds_tensor, self.test_labels_tensor)
        self.test_auc.update(self.test_preds_tensor, self.test_labels_tensor)

        pl_module.log("test/auroc", self.test_auroc.compute())
        pl_module.log("test/auc", self.test_auc.compute())

    def on_test_end(self, trainer, pl_module):
        # Create normalized confusion matrix
        conf_mat_norm = confusion_matrix(self.test_labels, self.test_preds, normalize="true")
        conf_mat_img_norm = ConfusionMatrixDisplay(
            conf_mat_norm, display_labels=trainer.datamodule.dataset_idx_to_class.values()
        )
        conf_mat_img_norm = self._confusion_matrix_as_pil(conf_mat_img_norm, "Normalized Confusion Matrix")

        # Create non-normalized confusion matrix
        conf_mat = confusion_matrix(self.test_labels, self.test_preds, normalize=None)
        conf_mat_img = ConfusionMatrixDisplay(conf_mat, display_labels=trainer.datamodule.dataset_idx_to_class.values())
        conf_mat_img = self._confusion_matrix_as_pil(conf_mat_img, "Confusion Matrix")

        wandb.log(
            {
                "Normalized Confusion Matrix": wandb.Image(conf_mat_img_norm),
                "Confusion Matrix": wandb.Image(conf_mat_img),
            }
        )
        wandb.save("normalized_confusion_matrix.jpg")
        wandb.save("confusion_matrix.jpg")

        exp_dir = get_exp_dir(trainer.logger)
        resume_ckpt_fpath = trainer.ckpt_path
        epoch = os.path.splitext(os.path.basename(resume_ckpt_fpath))[0].split("_")[1]
        exp_out_dir = os.path.join(exp_dir, "test", epoch)
        os.makedirs(exp_out_dir, exist_ok=True)

        for (label, pred), image_tensors in self.label_pred_to_image.items():
            label_name = trainer.datamodule.dataset_idx_to_class[label]
            pred_name = trainer.datamodule.dataset_idx_to_class[pred]
            out_dir = os.path.join(exp_out_dir, f"label={label_name}_pred={pred_name}")
            os.makedirs(out_dir, exist_ok=True)
            for i, image_tensor in tqdm(
                enumerate(image_tensors), desc=f"Saving images for label={label_name}_pred={pred_name}"
            ):
                image_pil = image_tensor_to_pil(image_tensor)
                image_pil.save(os.path.join(out_dir, f"{i}.jpg"))

        conf_mat_img_norm.save(os.path.join(exp_out_dir, "normalized_confusion_matrix.jpg"))
        conf_mat_img.save(os.path.join(exp_out_dir, "confusion_matrix.jpg"))

    def _confusion_matrix_as_pil(self, conf_mat_img, title):
        with NamedTemporaryFile(suffix=".jpg") as tmp:
            _, ax = plt.subplots(figsize=(10, 8))  # Set the figure size to 10x8 inches
            conf_mat_img.plot(ax=ax)
            plt.title(title)
            plt.setp(ax.get_xticklabels(), rotation=45)  # Rotate x-axis labels by 45 degrees
            plt.tight_layout()  # Adjust layout to prevent cutoff
            plt.savefig(tmp.name)
            img = Image.open(tmp.name)
        return img
