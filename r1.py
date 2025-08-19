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

# Regex patterns for numbers
number_patterns = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "DATE": r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
    "TAX_ID": r"\b\d{2}-\d{7}\b",
    "PHONE": r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b",
    "MONEY": r"\$\d+(?:,\d{3})*(?:\.\d{2})?"
}

# Other patterns (non-numeric sensitive info)
text_patterns = {
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
}

# Loop through detections
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = img[y1:y2, x1:x2]

        # OCR on detected region
        text = pytesseract.image_to_string(roi).strip()

        masked = False

        # Check number-based sensitive info
        for label, pattern in number_patterns.items():
            if re.search(pattern, text):
                # Replace with XXXX
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)  # white background
                cv2.putText(img, "XXXX", (x1 + 10, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                print(f"Replaced {label}: {text} → XXXX")
                masked = True
                break

        # Check text-based sensitive info (mask fully)
        if not masked:
            for label, pattern in text_patterns.items():
                if re.search(pattern, text):
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)  # Black box
                    print(f"Masked {label}: {text}")
                    masked = True
                    break

        # If no regex match but YOLO still marked it sensitive → mask with black box
        if not masked:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

# Save masked output
output_path = r"C:\Users\nevil\Desktop\C3\masked_mixed.png"
cv2.imwrite(output_path, img)
print(f"Masked image saved at: {output_path}")
