import cv2
import os
import albumentations as A
from glob import glob

input_dir = r"C:\Users\nevil\Desktop\C3\Dataset"    
output_dir = r"C:\Users\nevil\Desktop\C3\augmented_images"
os.makedirs(output_dir, exist_ok=True)

transform = A.Compose([
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.ISONoise(p=0.2),
    A.RandomScale(scale_limit=0.1, p=0.4),
    A.CoarseDropout(max_holes=3, max_height=20, max_width=20, fill_value=255, p=0.3),
    A.ImageCompression(quality_lower=30, quality_upper=70, p=0.4)
])

image_paths = []
for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):
    image_paths.extend(glob(os.path.join(input_dir, ext)))

image_paths = image_paths[:25]  
print(f"[INFO] Found {len(image_paths)} images")

num_augments = 5   

for idx, img_path in enumerate(image_paths):
    image = cv2.imread(img_path)
    if image is None:
        print(f"[WARN] Could not read {img_path}, skipping...")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for j in range(num_augments):
        augmented = transform(image=image)
        aug_img = augmented["image"]

        save_path = os.path.join(output_dir, f"aug_{idx+1}_{j+1}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
print(f"[SAVED] {save_path}")
