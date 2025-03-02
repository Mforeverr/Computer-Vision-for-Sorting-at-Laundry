import json
import os

def update_coco_file_names(json_file, old_base_dir, new_base_dir):
    # Baca file JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Perbarui path file_name untuk setiap gambar
    for image in data['images']:
        old_path = image['file_name']
        file_name = os.path.basename(old_path)  # Ambil nama file saja
        new_path = os.path.join(new_base_dir, file_name).replace("\\", "/")
        image['file_name'] = new_path

    # Simpan file JSON yang diperbarui
    output_file = os.path.splitext(json_file)[0] + "_updated.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"File JSON diperbarui dan disimpan sebagai: {output_file}")

# Contoh penggunaan
old_base_dir = r'train'  # Struktur lama (misalnya: train/location_6/shorts)
new_base_dir = r'images/train'  # Struktur baru (folder terpusat)

train_json = r'C:\trialopencv\aRTFClothes\annotations\train_annotations.json'
test_json = r'C:\trialopencv\aRTFClothes\annotations\test_annotations.json'

update_coco_file_names(train_json, old_base_dir, new_base_dir)
update_coco_file_names(test_json, old_base_dir, new_base_dir)
