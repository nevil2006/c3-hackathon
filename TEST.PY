import pytesseract
import cv2
from ultralytics import YOLO

model = YOLO(r"C:\Users\nevil\Desktop\C3\best (9).pt")
img = cv2.imread(r"C:\Users\nevil\Desktop\C3\38551989.png")

results = model(img)
data = {}

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # crop region
        roi = img[y1:y2, x1:x2]
        text = pytesseract.image_to_string(roi).strip()

        # store label + OCR text
        data[label] = text

# Example "user query"
def get_info(label):
    return data.get(label, "Not found")

print("Account Number:", get_info("Bank_Info"))
