import json
import os

def validate_coco_annotations(json_file, image_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Cek bagian images
    if 'images' not in data or not isinstance(data['images'], list):
        print("Error: Bagian 'images' tidak ditemukan atau bukan list.")
        return False

    # Cek bagian annotations
    if 'annotations' not in data or not isinstance(data['annotations'], list):
        print("Error: Bagian 'annotations' tidak ditemukan atau bukan list.")
        return False

    # Cek bagian categories
    if 'categories' not in data or not isinstance(data['categories'], list):
        print("Error: Bagian 'categories' tidak ditemukan atau bukan list.")
        return False

    # Verifikasi keberadaan gambar
    missing_images = []
    for image in data['images']:
        image_path = os.path.join(image_dir, image['file_name'])
        if not os.path.exists(image_path):
            missing_images.append(image['file_name'])

    if missing_images:
        print(f"Error: Gambar berikut tidak ditemukan: {missing_images}")
        return False

    # Verifikasi ID unik untuk images dan annotations
    image_ids = [img['id'] for img in data['images']]
    annotation_ids = [ann['id'] for ann in data['annotations']]

    if len(image_ids) != len(set(image_ids)):
        print("Error: ID gambar tidak unik.")
        return False

    if len(annotation_ids) != len(set(annotation_ids)):
        print("Error: ID anotasi tidak unik.")
        return False

    print("Validasi berhasil! Dataset tampaknya lengkap.")
    return True

# Contoh penggunaan
train_json = r'C:\trialopencv\ARTFClothes\train_annotations.json'
val_json = r'C:\trialopencv\ARTFClothes\val_annotations.json'
image_dir = r'C:\trialopencv\ARTFClothes\images\train'

validate_coco_annotations(train_json, image_dir)
validate_coco_annotations(val_json, image_dir)
