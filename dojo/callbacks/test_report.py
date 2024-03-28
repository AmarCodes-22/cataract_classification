from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import wandb


class GenerateTestReportCallback(Callback):
    def __init__(self):
        self.test_preds = list()
        self.test_labels = list()

        print("Initialized GenerateTestReportCallback")

    def on_test_batch_end(self, trainer, pl_module, output, batch, batch_idx, dataloader_idx=0):
        self.test_preds.extend(output["preds"].cpu().numpy())
        self.test_labels.extend(output["labels"].cpu().numpy())

    def on_test_end(self, trainer, pl_module):
        conf_mat = confusion_matrix(self.test_labels, self.test_preds)
        conf_mat_img = ConfusionMatrixDisplay(conf_mat, display_labels=trainer.datamodule.dataset_idx_to_class.values())
        conf_mat_img = self._confusion_matrix_as_pil(conf_mat_img)
        wandb.log({"Confusion Matrix": wandb.Image(conf_mat_img)})
        wandb.save("confusion_matrix.jpg")

    def _confusion_matrix_as_pil(self, conf_mat_img):
        with NamedTemporaryFile(suffix=".jpg") as tmp:
            _, ax = plt.subplots()
            conf_mat_img.plot(ax=ax)
            plt.savefig(tmp.name)
            img = Image.open(tmp.name)
        return img
