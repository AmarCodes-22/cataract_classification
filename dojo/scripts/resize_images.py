import os

import albumentations as A
import numpy as np
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def resize_image(image, max_dim):
    h, w = image.shape[:2]
    if h > w:
        new_h, new_w = max_dim, int(w * max_dim / h)
    else:
        new_h, new_w = int(h * max_dim / w), max_dim

    return np.array(Image.fromarray(image).resize((new_w, new_h)))


if __name__ == "__main__":
    images_dir = "/home/ubuntu/members/amar/projects/auto-classification-interior_subroi/data/v1"
    output_dir = "/home/ubuntu/members/amar/projects/auto-classification-interior_subroi/data/v1-small"

    os.makedirs(output_dir, exist_ok=True)

    # transform = A.Compose([A.SmallestMaxSize(max_size=1600)])
    transform = A.Compose([A.LongestMaxSize(max_size=512)])

    for root, _, fnames in os.walk(images_dir):
        for fname in tqdm(fnames, desc=f"Resizing images in {os.path.relpath(root, os.getcwd())}"):
            fpath = os.path.join(root, fname)
            try:
                img = np.array(Image.open(fpath))
                img = transform(image=img)["image"]
                save_fpath = fpath.replace(images_dir, output_dir)

                if os.path.exists(save_fpath):
                    continue

                os.makedirs(os.path.dirname(save_fpath), exist_ok=True)
                Image.fromarray(img).save(save_fpath)
            except Exception as e:
                print(e)
                print(fname)
