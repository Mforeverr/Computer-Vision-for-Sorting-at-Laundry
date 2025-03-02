import json
import os
from pathlib import Path

def coco_to_yolo(json_file, image_dir, output_dir):
    # Pastikan direktori output ada
    os.makedirs(output_dir, exist_ok=True)

    # Baca file JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Mapping kategori
    categories = {cat['id']: idx for idx, cat in enumerate(data['categories'])}

    # Proses setiap gambar
    for image_info in data['images']:
        image_id = image_info['id']
        image_width = image_info['width']
        image_height = image_info['height']
        image_filename = Path(image_info['file_name']).name  # Nama file gambar

        # Cari semua anotasi untuk gambar ini
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]

        # Jika tidak ada anotasi, lewati gambar
        if not annotations:
            continue

        # Buat file .txt untuk gambar ini
        txt_filename = Path(image_filename).with_suffix('.txt')
        with open(os.path.join(output_dir, txt_filename), 'w') as f:
            for ann in annotations:
                bbox = ann['bbox']  # Format COCO: [x_min, y_min, width, height]
                x_min, y_min, bbox_width, bbox_height = bbox

                # Konversi ke format YOLO
                x_center = (x_min + bbox_width / 2) / image_width
                y_center = (y_min + bbox_height / 2) / image_height
                width = bbox_width / image_width
                height = bbox_height / image_height

                # Ambil class ID
                category_id = ann['category_id']
                class_id = categories[category_id]

                # Tulis ke file .txt
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Contoh penggunaan
train_json = r'C:\trialopencv\aRTFClothes\annotations\train_annotations_updated.json'
test_json = r'C:\trialopencv\aRTFClothes\annotations\test_annotations_updated.json'

output_train_dir = r'C:\trialopencv\aRTFClothes\labels\train'
output_test_dir = r'C:\trialopencv\aRTFClothes\labels\test'

coco_to_yolo(train_json, r'C:\trialopencv\aRTFClothes\images\train', output_train_dir)
coco_to_yolo(test_json, r'C:\trialopencv\aRTFClothes\images\test', output_test_dir)
