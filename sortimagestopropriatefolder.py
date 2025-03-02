import os
import shutil

def merge_images(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_dir, file)
                shutil.copy(src_path, dst_path)

# Contoh penggunaan
source_train_dir = r'C:\trialopencv\aRTFClothes\train'
source_test_dir = r'C:\trialopencv\aRTFClothes\test'

target_train_dir = r'C:\trialopencv\aRTFClothes\images\train'
target_test_dir = r'C:\trialopencv\aRTFClothes\images\test'

merge_images(source_train_dir, target_train_dir)
merge_images(source_test_dir, target_test_dir)
