import cv2
from ultralytics import YOLO

sensitive_classes = [
    "Bank_Info", "BARCODE", "Company_Info", "Contact Information",
    "DATES", "Invoice_Number", "Monetary_Value", "Other_Codes",
    "REMIT To", "SHIPING ADDRESS", "Shipment Details", "Tax_IDs"
]

model = YOLO(r"C:\Users\nevil\Desktop\C3\best (9).pt")

img_path = r"C:\Users\nevil\Desktop\C3\38551989.png"
img = cv2.imread(img_path)


results = model(img)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label in sensitive_classes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)  

output_path = r"C:\Users\nevil\Desktop\C3\masked.png"
cv2.imwrite(output_path, img)
print(f" Masked image saved at: {output_path}")
