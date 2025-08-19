# mask_sensitive_with_spacy.py
import cv2
import re
import json
import numpy as np
from ultralytics import YOLO

import pytesseract
from pytesseract import Output

import spacy
# If on Windows, set tesseract path (uncomment & edit)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------
# Config / paths (edit these)
# -----------------------
YOLO_MODEL_PATH = r"C:\Users\nevil\Desktop\C3\best (9).pt"
IMAGE_PATH = r"C:\Users\nevil\Desktop\C3\38551989.png"
OUTPUT_PATH = r"C:\Users\nevil\Desktop\C3\masked_spacy.png1"

# choose mask method: "black" | "blur" | "pixelate" | "watermark"
MASK_METHOD = "black"

# YOLO sensitive label names (as in your model)
SENSITIVE_CLASSES = [
    "Bank_Info", "BARCODE", "Company_Info", "Contact Information",
    "DATES", "Invoice_Number", "Monetary_Value", "Other_Codes",
    "REMIT To", "SHIPING ADDRESS", "Shipment Details", "Tax_IDs"
]

# Regex patterns
PATTERNS = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "SSN_nodash": re.compile(r"\b\d{9}\b"),
    "DATE_mmddyyyy": re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
    "DATE_iso": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "ACCOUNT_8_12": re.compile(r"\b\d{8,12}\b"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "IBAN_SIMPLE": re.compile(r"\b[A-Z]{2}\d{2}[A-Za-z0-9]{10,30}\b"),
}

MAX_WINDOW = 4  # sliding window for OCR token joining

# -----------------------
# spaCy NER model
# -----------------------
nlp = spacy.load("en_core_web_sm")  # you confirmed this is installed

# -----------------------
# Masking utilities
# -----------------------
def mask_black(img, box):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,0), thickness=-1)

def mask_blur(img, box, ksize=51, sigma=30):
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return
    k = min(ksize, max(3, (min(roi.shape[:2])//2)*2+1))
    blurred = cv2.GaussianBlur(roi, (k,k), sigma)
    img[y1:y2, x1:x2] = blurred

def mask_pixelate(img, box, blocks=10):
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, blocks), max(1, int(blocks*h/w))), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = pix

def mask_watermark(img, box, text="REDACTED"):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), thickness=-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, (x2-x1)/300)
    thickness = 2
    ts = cv2.getTextSize(text, font, font_scale, thickness)[0]
    tx = x1 + max(0, (x2-x1 - ts[0])//2)
    ty = y1 + max(ts[1], (y2-y1 + ts[1])//2)
    cv2.putText(img, text, (tx, ty), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)

MASK_METHODS = {
    "black": mask_black,
    "blur": mask_blur,
    "pixelate": mask_pixelate,
    "watermark": mask_watermark
}

def clamp(v, lo, hi):
    return max(lo, min(int(v), hi))

# -----------------------
# OCR tokenization helper (returns word entries)
# -----------------------
def get_ocr_word_entries(img):
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    n = len(data['text'])
    entries = []
    for i in range(n):
        text = data['text'][i].strip()
        if text == "":
            continue
        left, top, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        entries.append({
            "text": text,
            "left": left,
            "top": top,
            "right": left + width,
            "bottom": top + height,
            "width": width,
            "height": height,
            "idx": i
        })
    return entries

# -----------------------
# Use pytesseract tokens + regex to find sensitive boxes
# -----------------------
def ocr_regex_boxes(img):
    h, w = img.shape[:2]
    entries = get_ocr_word_entries(img)
    boxes = []

    for i in range(len(entries)):
        for window in range(1, MAX_WINDOW+1):
            j = i + window
            if j > len(entries): break
            tokens = [entries[k]["text"] for k in range(i, j)]
            cand_space = " ".join(tokens)
            cand_nospace = "".join(tokens)
            cand_dash = "-".join(tokens)
            candidates = [cand_space, cand_nospace, cand_dash]
            for cand in candidates:
                for label, patt in PATTERNS.items():
                    if patt.search(cand):
                        left = min(entries[k]["left"] for k in range(i, j))
                        top  = min(entries[k]["top"] for k in range(i, j))
                        right= max(entries[k]["right"] for k in range(i, j))
                        bottom=max(entries[k]["bottom"] for k in range(i, j))
                        left = clamp(left, 0, w-1); top = clamp(top, 0, h-1)
                        right = clamp(right, 0, w-1); bottom = clamp(bottom, 0, h-1)
                        boxes.append({"box": (left, top, right, bottom), "label": label, "text": cand})
                        break
                else:
                    continue
                break
    return boxes

# -----------------------
# Use spaCy NER on full OCR text and map to token boxes
# -----------------------
def spacy_ner_boxes(img):
    h, w = img.shape[:2]
    entries = get_ocr_word_entries(img)
    # build full OCR text with tokens separated by space and also store token indices to recover positions
    token_texts = [e["text"] for e in entries]
    full_text = " ".join(token_texts)
    doc = nlp(full_text)
    boxes = []
    # For each entity detected by spaCy, try to find matching sequence in OCR tokens
    for ent in doc.ents:
        if ent.label_ in {"PERSON","ORG","GPE","LOC","MONEY","DATE","CARDINAL","NORP","FAC"}:
            ent_text = ent.text.strip()
            # try to find ent_text in token list with sliding windows (normalize simple whitespace)
            tokens = token_texts
            found = False
            for i in range(len(tokens)):
                for window in range(1, MAX_WINDOW+5):
                    j = i + window
                    if j > len(tokens): break
                    cand = " ".join(tokens[i:j])
                    if cand.lower() == ent_text.lower():
                        left = min(entries[k]["left"] for k in range(i, j))
                        top  = min(entries[k]["top"] for k in range(i, j))
                        right= max(entries[k]["right"] for k in range(i, j))
                        bottom=max(entries[k]["bottom"] for k in range(i, j))
                        left = clamp(left, 0, w-1); top = clamp(top, 0, h-1)
                        right = clamp(right, 0, w-1); bottom = clamp(bottom, 0, h-1)
                        boxes.append({"box": (left, top, right, bottom), "label": f"SPACY_{ent.label_}", "text": ent_text})
                        found = True
                        break
                if found:
                    break
    return boxes

# -----------------------
# YOLO detection wrapper
# -----------------------
def yolo_boxes_from_model(img, model_path):
    model = YOLO(model_path)
    results = model(img)
    boxes = []
    for r in results:
        for box in r.boxes:
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coords)
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label in SENSITIVE_CLASSES:
                boxes.append({"box": (x1,y1,x2,y2), "label": f"YOLO_{label}", "score": float(box.conf[0]) if hasattr(box, 'conf') else None})
    return boxes

# -----------------------
# Combine boxes (simple union of overlapping boxes)
# -----------------------
def combine_boxes(detections):
    final = []
    for d in detections:
        box = d["box"]
        merged = False
        for i, fb in enumerate(final):
            # check overlap
            ix = max(0, min(box[2], fb[2]) - max(box[0], fb[0]))
            iy = max(0, min(box[3], fb[3]) - max(box[1], fb[1]))
            if ix*iy > 0:  # intersect area > 0
                final[i] = (min(fb[0], box[0]), min(fb[1], box[1]), max(fb[2], box[2]), max(fb[3], box[3]))
                merged = True
                break
        if not merged:
            final.append(box)
    return final

# -----------------------
# Main pipeline
# -----------------------
def run_masking(image_path, model_path, output_path, mask_method="black"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape[:2]
    audit = {"image": image_path, "detections": []}

    # YOLO
    try:
        yolo_dets = yolo_boxes_from_model(img, model_path)
        audit["detections"].extend(yolo_dets)
    except Exception as e:
        print("YOLO warning:", e)
        yolo_dets = []

    # OCR + regex
    ocr_dets = ocr_regex_boxes(img)
    audit["detections"].extend([{"box": d["box"], "label": "OCR_REGEX_"+d["label"], "text": d["text"]} for d in ocr_dets])

    # spaCy NER
    spacy_dets = spacy_ner_boxes(img)
    audit["detections"].extend(spacy_dets)

    # combine to final boxes
    final_boxes = combine_boxes(audit["detections"])

    # apply chosen mask
    method_fn = MASK_METHODS.get(mask_method, mask_black)
    for b in final_boxes:
        x1,y1,x2,y2 = b
        x1 = clamp(x1, 0, w-1); y1 = clamp(y1, 0, h-1)
        x2 = clamp(x2, 0, w-1); y2 = clamp(y2, 0, h-1)
        method_fn(img, (x1,y1,x2,y2))

    cv2.imwrite(output_path, img)
    print("Saved:", output_path)
    audit_path = output_path.rsplit(".",1)[0] + "_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print("Audit saved:", audit_path)
    return output_path, audit

# -----------------------
# Run as script
# -----------------------
if __name__ == "__main__":
    out, audit = run_masking(IMAGE_PATH, YOLO_MODEL_PATH, OUTPUT_PATH, mask_method=MASK_METHOD)
