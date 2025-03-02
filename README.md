
# **Computer Vision for Laundry Sorting**

## **Project Description**
This project is an implementation of **object detection using the YOLOv8 model** for **computer vision** applications in clothing categories. The main goal of this project is to build an MVP (Minimum Viable Product) system that can detect and classify clothing categories in real-time during the sorting process.

However, evaluation results show that detection accuracy is still suboptimal because the training dataset does not fully match the actual environment where the application will be used. Therefore, a **new dataset more relevant to the actual environment** is being collected to improve model performance.

The dataset currently in use comes from the following repository:
- **Dataset Source**: [aRTF-Clothes-dataset](https://github.com/tlpss/aRTF-Clothes-dataset/tree/main)

The labels in the dataset remain as provided by the **aRTF author**, without relabeling. Only dataset separation and conversion were performed to make it compatible with YOLOv8.

For more information about this dataset, please refer to the following paper:

```bibtex
@misc{lips2024learning,
      title={Learning Keypoints for Robotic Cloth Manipulation using Synthetic Data},
      author={Thomas Lips and Victor-Louis De Gusseme and Francis wyffels},
      year={2024},
      eprint={2401.01734},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

---

## **Project Objectives**
1. **Real-Time Clothing Category Detection**:
   - Assisting in the sorting process by automatically detecting and classifying clothing categories.
   - Supporting input from webcams or other video sources for direct application in industrial or warehouse environments.

2. **Improving Sorting Efficiency**:
   - Reducing reliance on human labor in the clothing category identification process.
   - Ensuring classification accuracy to avoid sorting errors.

3. **Exploration of Computer Vision**:
   - Demonstrating the potential of computer vision technology in industrial applications such as inventory management and logistics.

---

## **Hardware Specifications**
This project was tested using the following hardware:
- **CPU**: Intel Core i5-10850H
- **GPU**: NVIDIA Quadro RTX 3000 (6GB VRAM)
- **RAM**: 64GB
- **Storage**: SSD (to speed up the training and inference process)

### **Training Time**
- Fine-tuning the **YOLOv8n** model with **150 epochs** took approximately **3 hours** using the hardware above.

---

## **Key Features**
1. **Real-Time Object Detection**:
   - Supports input from webcams, video files, and images.
   - Capable of detecting objects in real-time with decent performance.

2. **Model Fine-Tuning**:
   - The YOLOv8 model has been fine-tuned using a custom dataset to improve accuracy for specific classes.

3. **Compatibility**:
   - Can run in a local environment with GPU support for accelerated inference.

4. **Ease of Use**:
   - The project is designed to be easy to use, both through the CLI (Command Line Interface) and directly via Python code.

---

## **What Has Been Done**
The following is a list of activities that have been successfully completed in this project:

### **1. Dataset Preparation**
- **Data Collection**: The initial dataset was taken from [aRTF-Clothes-dataset](https://github.com/tlpss/aRTF-Clothes-dataset/tree/main).
- **Data Labeling**: The dataset was labeled using tools like [LabelImg](https://github.com/heartexlabs/labelImg) or [CVAT](https://cvat.org/).
- **Dataset Format Conversion**: The dataset was converted into a format compatible with YOLOv8 (`.txt` format for bounding boxes and `.yaml` for configuration).
- **Dataset Re-Sorting**:
  - Sorted the dataset by class to ensure balanced data distribution.
  - Removed images without labels or invalid bounding boxes.
  - Used data augmentation techniques to increase dataset diversity.

### **2. Fine-Tuning the YOLOv8 Model**
- Used a custom dataset to fine-tune the YOLOv8 model.
- Training was conducted with the following configuration:
  - Epochs: **150**
  - Image Size: **640x640**
  - Batch Size: **16**
  - Optimizer: **SGD**
- Evaluated the model using metrics such as mAP (mean Average Precision), precision, and recall.

### **3. Real-Time Detection Implementation**
- Successfully integrated the model into a real-time application using OpenCV.
- Supports input from a webcam (`source=0`), video files, and images.
- Added the `cv2.CAP_DSHOW` backend for compatibility with Windows systems.

### **4. Testing and Validation**
- Conducted testing on various types of input (webcam, video, images).
- Ensured the model could detect objects with adequate accuracy.
- Identified areas that need improvement (see the "Areas for Improvement" section).

---

## **Achieved Results**
- **Detection Accuracy**: The model successfully detected objects with fairly high accuracy on the custom dataset.
- **Real-Time Performance**: Inference ran smoothly on the GPU, with a frame rate sufficient for real-time applications.
- **Ease of Integration**: The project can easily be integrated into other applications or run independently.

---

## **Conclusion**
Evaluation results show that model performance heavily depends on the **relevance of the dataset to the actual environment**. The dataset currently in use ([aRTF-Clothes-dataset](https://github.com/tlpss/aRTF-Clothes-dataset/tree/main)) has backgrounds and lighting conditions that differ from the actual environment where the application will be deployed. As a result, the model struggles to detect objects with optimal accuracy.

To improve model performance, the following steps will be taken:
1. **Collecting a New Dataset**:
   - A new dataset will be collected in the actual environment to ensure that the background, lighting, and other conditions match real-world usage.
2. **Adding Data Augmentation**:
   - Data augmentation techniques such as rotation, flipping, and brightness adjustment will be used to increase dataset diversity.
3. **Re-Fine-Tuning the Model**:
   - The model will be fine-tuned again using the new dataset to improve detection accuracy.

---

## **Areas for Improvement**
Although this project has achieved its MVP goals, there are still several areas that need improvement or further development:

### **1. Detection Accuracy**
- Some classes are often misdetected (e.g., objects are frequently detected as "shorts").
- Solutions:
  - Collect a new dataset more relevant to the actual environment.
  - Add more training data for underrepresented classes.

### **2. Performance at Long Distances**
- Objects far from the camera are often not detected well.
- Solutions:
  - Fine-tune the model with a dataset that includes objects at various distances.
  - Use a camera with optical zoom to improve long-distance detection capabilities.

### **3. Performance Optimization**
- Inference on the CPU is still slow for real-time applications.
- Solutions:
  - Optimize the model using techniques like quantization or pruning.
  - Use more powerful hardware (e.g., NVIDIA GPU with CUDA).

### **4. Documentation and Usage**
- Current documentation is limited.
- Solutions:
  - Add step-by-step installation and usage tutorials.
  - Include example videos or GIFs to demonstrate detection results.

---

## **How to Run the Project**
The following are the steps to run this project:

### **1. Install Dependencies**
Install all required libraries:
```bash
pip install ultralytics opencv-python
```

### **2. Dataset Conversion and Sorting**
Before training the model, ensure the dataset is prepared correctly:
1. **Dataset Format Conversion**:
   - Ensure the dataset is in YOLOv8 format (bounding boxes in `.txt` format and configuration in `.yaml` format).
   - Use the following script for conversion if needed:
     ```python
     from ultralytics.utils import convert_coco_to_yolo

     convert_coco_to_yolo("path/to/coco_dataset", "path/to/output_yolo")
     ```

2. **Dataset Sorting**:
   - Split the dataset into `train`, `val`, and `test` folders.
   - Ensure class distribution is balanced. Use the following script to check the distribution:
     ```python
     from collections import Counter
     import os

     def count_classes(label_dir):
         class_counts = Counter()
         for label_file in os.listdir(label_dir):
             with open(os.path.join(label_dir, label_file), "r") as f:
                 lines = f.readlines()
                 classes = [line.split()[0] for line in lines]
                 class_counts.update(classes)
         return class_counts

     print(count_classes("path/to/labels"))
     ```

3. **Data Augmentation**:
   - Use data augmentation to increase dataset diversity:
     ```python
     from ultralytics.data.augment import augment_image

     augmented_image = augment_image(original_image, flip=True, rotate=30)
     ```

### **3. Fine-Tuning the Model**
Run fine-tuning using the custom dataset:
```bash
yolo train model=yolov8n.pt data=path/to/data.yaml epochs=150 imgsz=640
```

### **4. Real-Time Inference**
Run real-time object detection using a webcam:
```bash
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=0
```

Or use the following Python code directly:
```python
from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## **References**
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- Labeling Tools: [LabelImg](https://github.com/heartexlabs/labelImg), [CVAT](https://cvat.org/)
- Dataset: [aRTF-Clothes-dataset](https://github.com/tlpss/aRTF-Clothes-dataset/tree/main)

---

## **Contributions**
If you wish to contribute to this project, please open an issue or pull request in this repository.

---

## Lisensi
Proyek ini dilisensikan di bawah [GPL-3.0 License](LICENSE)..
