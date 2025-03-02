from ultralytics import YOLO
import cv2

# Load model YOLOv8
model_path = "runs/detect/train9/weights/best.pt"  # Ganti dengan path ke model Anda
model = YOLO(model_path)

# Buka webcam (gunakan source=0 untuk webcam)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Gunakan backend DirectShow untuk Windows

if not cap.isOpened():
    print("Gagal membuka kamera.")
    exit()

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Lakukan inferensi menggunakan model YOLOv8
    results = model(frame)  # Inferensi pada frame

    # Proses hasil deteksi
    annotated_frame = results[0].plot()  # Gambar bounding box dan label pada frame

    # Tampilkan hasil di jendela GUI
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
