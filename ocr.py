import cv2
import pytesseract
import csv
from ultralytics import YOLO

model = YOLO(r"C:\Users\nevil\Desktop\C3\best (9).pt")

img_path = r"C:\Users\nevil\Desktop\C3\38551989.png"
img = cv2.imread(img_path)

results = model(img)

csv_path = r"C:\Users\nevil\Desktop\C3\ocr_output.csv"
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Confidence", "Extracted_Text"])

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            roi = img[y1:y2, x1:x2]

            text = pytesseract.image_to_string(roi).strip()

            writer.writerow([label, f"{conf:.2f}", text])

print(f" OCR results saved to {csv_path}")
