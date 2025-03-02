import os
import json

def merge_coco_annotations(json_files, output_file):
    # Pastikan direktori induk ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    merged_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id_offset = 0
    annotation_id_offset = 0

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Tambahkan gambar
        for image in data['images']:
            image['id'] += image_id_offset
            merged_data['images'].append(image)

        # Tambahkan anotasi
        for annotation in data['annotations']:
            annotation['id'] += annotation_id_offset
            annotation['image_id'] += image_id_offset
            merged_data['annotations'].append(annotation)

        # Tambahkan kategori (jika belum ada)
        for category in data['categories']:
            if category not in merged_data['categories']:
                merged_data['categories'].append(category)

        # Update offset
        image_id_offset += len(data['images'])
        annotation_id_offset += len(data['annotations'])

    # Simpan file JSON gabungan
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)

# Contoh penggunaan
train_json_files = [
    r'C:\trialopencv\ARTFClothes\shorts-train.json',
    r'C:\trialopencv\ARTFClothes\towels-train.json',
    r'C:\trialopencv\ARTFClothes\tshirts-train.json'
]
merge_coco_annotations(train_json_files, r'C:\trialopencv\aRTFClothes\annotations\train_annotations.json')

test_json_files = [
    r'C:\trialopencv\ARTFClothes\shorts-test.json',
    r'C:\trialopencv\ARTFClothes\towels-test.json',
    r'C:\trialopencv\ARTFClothes\tshirts-test.json'
]
merge_coco_annotations(test_json_files, r'C:\trialopencv\aRTFClothes\annotations\test_annotations.json')
