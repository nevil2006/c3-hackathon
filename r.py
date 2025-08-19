import cv2
import pytesseract
import re
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO(r"C:\Users\nevil\Desktop\C3\best (9).pt")

# Input image
img_path = r"C:\Users\nevil\Desktop\C3\input image.png"
img = cv2.imread(img_path)

# Run YOLO detection
results = model(img)

# Define regex patterns for sensitive info
patterns = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "DATE": r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
    "TAX_ID": r"\b\d{2}-\d{7}\b",
    "PHONE": r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b",
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "MONEY": r"\$\d+(?:,\d{3})*(?:\.\d{2})?"
}

# Loop through detections
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop detected region
        roi = img[y1:y2, x1:x2]

        # OCR on the cropped region
        text = pytesseract.image_to_string(roi)

        # Check regex patterns
        for label, pattern in patterns.items():
            if re.search(pattern, text):
                # Mask the region if regex confirms sensitive data
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
                print(f"Masked {label}: {text.strip()}")
                break

# Save masked output
output_path = r"C:\Users\nevil\Desktop\C3\masked_regex.png"
cv2.imwrite(output_path, img)
print(f"Masked image saved at: {output_path}")
