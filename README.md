# **C3 Hackathon** 

An AI-powered project developed for the **C3 Hackathon**.  
This repository combines **OCR (Optical Character Recognition)**, **sensitive data masking**, and **deep learning models** to process and secure text from images.  



##  Repository Structure  

```c3-hackathon/
│
├── Dataset/ # Raw dataset used for training
├── augmented_images/ # Data after augmentation
├── runs/detect/ # YOLO or model detection output
│
├── SAMPLE.PY # Sample script to test pipeline
├── TEST.PY # Testing script
├── mask.py # Script to mask sensitive information
├── mask_sensitive_with_spacy.py # Masking with NLP (spaCy)
├── ocr.py # OCR pipeline to extract text from images
├── r.py / r1.py # Additional helper/test scripts
│
├── best (9).pt # Trained PyTorch model
├── 38551989.png # Example input image
├── masked.png # Example masked output
├── masked_mixed.png
├── masked_regex.png
│
├── ocr_output.csv # OCR results stored as CSV``






## \ Features  

-  **OCR Processing** – Extract text from input images  
-  **Sensitive Data Masking** – Hide personal/sensitive data using **regex** and **NLP (spaCy)**  
-  **Deep Learning Model** – Pretrained **PyTorch model** (`best (9).pt`) included  
-  **CSV Export** – Save extracted text and results to `.csv`  
-  **Sample & Test Scripts** – Quick start with `SAMPLE.PY` and `TEST.PY`  

---

##  Installation  


# Clone the repository
```git clone https://github.com/nevil2006/c3-hackathon.git```

```cd c3-hackathon```

# Create a virtual environment (optional but recommended)
```python -m venv venv```

```source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows```

# Install dependencies
``pip install -r requirements.txt``


** Example Outputs**
Input Image → 38551989.png

OCR Result → ocr_output.csv

Masked Outputs →

masked.png

masked_mixed.png

masked_regex.png

**Tech Stack**

--Python 3.x

---PyTorch – Model training & inference

----OpenCV – Image preprocessing

---Tesseract OCR – Text extraction

---spaCy – NLP-based masking

---Regex – Rule-based sensitive info filtering

 Contributors
--Nevil J – AI & Data Science Student

--R.S. Hariharan – AI & Data Science Student

--Ajay – AI & Data Science Student

--Abinav Prakash – AI & Data Science Student

--Gunal P – AI & Data Science Student

