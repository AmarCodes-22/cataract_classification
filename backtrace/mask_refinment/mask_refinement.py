import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.insert(0, '/home/ubuntu/members/utkarsh/backtracing_pipeline/dojo/backtrace')
print(sys.path)
from utils import mask_process_interior
class InspyrenetSegmentationModel:
    def __init__(self, model_path: str, input_size: tuple = (768, 768)):
        self.model = torch.jit.load(model_path).cuda()
        self.input_size = input_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        original_height, original_width = image.shape[:2]
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        image = np.array(Image.fromarray(image).resize(self.input_size, Image.BILINEAR))
        image = torch.tensor(image).unsqueeze(0).float().cuda()
        
        output = self.model(image)
        output = output.squeeze().cpu().numpy()
        output = cv2.resize(output, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        return output
    
def is_black_image(image: np.ndarray) -> bool:
    return np.all(image == 0)

def process_images(input_folder, output_folder, model):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to load image: {input_path}")
                continue

            if is_black_image(image):
                print(f"Image is black, creating a black mask for: {input_path}")
                output = np.zeros_like(image)
            else:
                print(f"Processing image: {input_path}")
                output = model(image)
                output = mask_process_interior(output, image, apply_dilate=False)
            
            success = cv2.imwrite(output_path, output)
            if not success:
                print(f"Failed to save image: {output_path}")
            else:
                print(f"Image saved successfully: {output_path}")

if __name__ == "__main__":
    pass_dir = sys.argv[1]
    fail_dir = sys.argv[2]
    output_pass_dir = sys.argv[3]
    output_fail_dir = sys.argv[4]
    model_path = sys.argv[5]

    model = InspyrenetSegmentationModel(model_path=model_path)

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.submit(process_images, pass_dir, output_pass_dir, model)
        executor.submit(process_images, fail_dir, output_fail_dir, model)
