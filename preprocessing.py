import os, sys

root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(root))
raw_data_path = os.path.join(root, "ICU_data")
copy_dir_list = ["color_img", "depth_data"]
data_list = {"train": ["1", "2", "3", "4", "5", "6", "7"], "val": ["8"], "test": ["9", "10"]}

import shutil
from save_video import D2RGB
import numpy as np
import cv2
import pathlib


def compress_depth_map(dst):

    depth_data_folder = os.path.join(dst, "depth_data")
    rgb_data_folder = os.path.join(dst, "depth_map")

    for depth_data_file in os.listdir(depth_data_folder):
        depth_data = np.load(os.path.join(depth_data_folder, depth_data_file))
        rgb_data = D2RGB(depth_data)

        cv2.imwrite(
            os.path.join(rgb_data_folder, os.path.splitext(depth_data_file)[0] + ".png"), cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
        )


def move_images():

    for (key, dir_list) in data_list.items():
        dst = os.path.join(root, "data", key)
        global_idx = 0

        for dir in dir_list:
            dir_path = os.path.join(raw_data_path, dir)
            for angle in os.listdir(dir_path):
                angle_path = os.path.join(dir_path, angle)
                for movement in os.listdir(angle_path):
                    movement_path = os.path.join(angle_path, movement)
                    for data_type in copy_dir_list:
                        data_type_path = os.path.join(movement_path, data_type)
                        local_idx = global_idx
                        sorted_files = sorted(os.listdir(data_type_path), key=lambda x: int(pathlib.Path(x).stem))
                        for file in sorted_files:
                            _, ext = os.path.splitext(file)
                            src_path = os.path.join(movement_path, data_type, file)
                            dst_path = os.path.join(dst, data_type, str(local_idx).zfill(5) + ext)
                            shutil.copyfile(src_path, dst_path)
                            local_idx += 1
                    global_idx = local_idx

        compress_depth_map(dst)


if __name__ == "__main__":
    move_images()
    pass
