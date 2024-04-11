import os
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import wandb
from lightning.pytorch.callbacks import Callback
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm

from dojo.utils import get_exp_dir, image_tensor_to_pil


# todo: find the image size based on the number of classes and adjust the figure size accordingly
class GenerateTestReportCallback(Callback):
    def __init__(self):
        self.test_preds = list()
        self.test_labels = list()
        self.label_pred_to_image = dict()

        print("Initialized GenerateTestReportCallback")

    def on_test_batch_end(self, trainer, pl_module, output, batch, batch_idx, dataloader_idx=0):
        self.test_preds.extend(output["preds"].cpu().numpy())
        self.test_labels.extend(output["labels"].cpu().numpy())

        preds, labels = output["preds"].cpu().numpy(), output["labels"].cpu().numpy()
        mismatch_indices = np.where(np.not_equal(preds, labels))[0]
        for index in mismatch_indices:
            image_tensor = batch["image"][index]
            label = labels[index]
            prediction = preds[index]
            if (label, prediction) not in self.label_pred_to_image:
                self.label_pred_to_image[(label, prediction)] = []
            self.label_pred_to_image[(label, prediction)].append(image_tensor)

    def on_test_end(self, trainer, pl_module):
        conf_mat = confusion_matrix(self.test_labels, self.test_preds)
        conf_mat_img = ConfusionMatrixDisplay(conf_mat, display_labels=trainer.datamodule.dataset_idx_to_class.values())
        conf_mat_img = self._confusion_matrix_as_pil(conf_mat_img)
        wandb.log({"Confusion Matrix": wandb.Image(conf_mat_img)})
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

        conf_mat_img.save(os.path.join(exp_out_dir, "confusion_matrix.jpg"))

    def _confusion_matrix_as_pil(self, conf_mat_img):
        with NamedTemporaryFile(suffix=".jpg") as tmp:
            _, ax = plt.subplots(figsize=(10, 8))  # Set the figure size to 10x8 inches
            conf_mat_img.plot(ax=ax)
            plt.setp(ax.get_xticklabels(), rotation=45)  # Rotate x-axis labels by 45 degrees
            plt.savefig(tmp.name)
            img = Image.open(tmp.name)
        return img
