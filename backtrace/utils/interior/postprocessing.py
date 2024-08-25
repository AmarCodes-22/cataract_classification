import os
import shutil
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image
from skimage import exposure


def mask_process_interior(mask, ori, apply_dilate=False, rem_out=True):
    temp_path = "/tmp"
    mask = np.stack([mask, mask, mask], -1)
    uuid_prefix = str(uuid4())
    dir_path = os.path.join(temp_path, uuid_prefix)
    os.mkdir(dir_path)
    try:
        path = os.path.join(temp_path, uuid_prefix, uuid_prefix)
        ppm_name, svg_name, mask_name = (
            path + ".ppm",
            path + ".svg",
            path + "_mask.png",
        )

        size_ = (1000, 1000)
        mask = cv2.resize(mask, size_)

        Image.fromarray(mask).save(ppm_name)
        os.system(f"potrace -s {ppm_name} -o {svg_name}")
        os.system(f"convert {svg_name} {mask_name}")

        mask = cv2.imread(mask_name, 0)
        mask = 255 - mask
        ksz = int((min(size_) // 100) / 3)
        if ksz % 2 == 0:
            ksz -= 1

        if apply_dilate:
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz)))

        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.resize(mask, ori.shape[-2::-1], interpolation=cv2.INTER_CUBIC)

        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.5, sigmaY=1.5, borderType=cv2.BORDER_DEFAULT)
        mask = exposure.rescale_intensity(mask, in_range=(180, 255), out_range=(0, 255)).astype(np.uint8)

        # out = np.dstack((ori, mask))

    finally:
        shutil.rmtree(dir_path, ignore_errors=True)

    return np.array(mask)
