import numpy as np
import torch
from PIL import Image

# todo: move to datasets


def image_tensor_to_pil(image_tensor: torch.Tensor) -> Image:
    """Converts a torch.Tensor image to a PIL Image.

    Args:
        image_tensor (torch.Tensor): The image tensor.

    Returns:
        Image: The PIL Image.
    """
    image_tensor = image_tensor.cpu().numpy()
    image_tensor = np.transpose(image_tensor, (1, 2, 0))
    image_tensor = (image_tensor * 255).astype(np.uint8)
    return Image.fromarray(image_tensor)
