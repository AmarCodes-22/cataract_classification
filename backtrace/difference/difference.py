import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np


class BacktraceProcessor:
    def __init__(self):
        self.backtrace_dir = os.getenv("BACKTRACE_DIR")
        self.without_backtrace_dir = os.getenv("WITHOUT_BACKTRACE_DIR")
        self.ai_dash_dir = os.getenv("AI_DASH_DIR")
        self.manual_dir = os.getenv("MANUAL_DIR")
        self.diff_pass_dir = os.getenv("DIFFERENCE_PASS_DIR")
        self.diff_fail_dir = os.getenv("DIFFERENCE_FAIL_DIR")

    def create_directory(self, directory_name):
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

    def compare_images(self, ai_image, manual_image):
        path_result = os.path.join(
            self.diff_pass_dir if ai_image.startswith(self.backtrace_dir) else self.diff_fail_dir,
            os.path.basename(ai_image),
        )

        if not os.path.exists(path_result):

            image1 = cv2.imread(ai_image)
            image2 = cv2.imread(manual_image)

            if image1 is None or image2 is None:
                print(f"Error loading images {ai_image} or {manual_image}")
                return

            if image1.shape != image2.shape:
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

            diff_r = cv2.absdiff(image1[:, :, 0], image2[:, :, 0])
            diff_g = cv2.absdiff(image1[:, :, 1], image2[:, :, 1])
            diff_b = cv2.absdiff(image1[:, :, 2], image2[:, :, 2])

            diff = np.maximum(np.maximum(diff_r, diff_g), diff_b)

            diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)

            if np.median(diff) == 0:
                cv2.imwrite(path_result, diff)
                print(f"Processed and saved permanently: {path_result}")

    def process_images(self, images_dir1, images_dir2, diff_dir):
        images1 = set(os.listdir(images_dir1))
        images2 = set(os.listdir(images_dir2))

        common_images = images1.intersection(images2)
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(
                    self.compare_images, os.path.join(images_dir1, image_name), os.path.join(images_dir2, image_name)
                ): image_name
                for image_name in common_images
            }
            for future in as_completed(futures):
                future.result()

    def run(self):
        self.create_directory(self.diff_pass_dir)
        self.create_directory(self.diff_fail_dir)
        status = False
        while not status:
            with ThreadPoolExecutor(max_workers=4) as executor:
                executor.submit(self.process_images, self.backtrace_dir, self.without_backtrace_dir, self.diff_pass_dir)
                executor.submit(self.process_images, self.ai_dash_dir, self.manual_dir, self.diff_fail_dir)

            if os.path.exists("difference.flag"):
                status = True

        print("Bulk processing complete. Executing final checks...")
        self.final_check_and_process()
        with open("backtrace_complete.log", "w") as log:
            log.write("Backtrace processing completed successfully.")
        print("Final processing complete. Exiting...")

    def final_check_and_process(self):
        self.process_images(self.backtrace_dir, self.without_backtrace_dir, self.diff_pass_dir)
        self.process_images(self.ai_dash_dir, self.manual_dir, self.diff_fail_dir)


if __name__ == "__main__":
    processor = BacktraceProcessor()
    processor.run()
