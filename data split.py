import os
import shutil
import random

# ======= CHANGE THESE PATHS =======
source_dir = "fer2013/train"   # your current train folder
train_out = "fer2013/train_new"
val_out = "fer2013/val_new"

split_ratio = 0.9  # 90% train, 10% val
# =================================

os.makedirs(train_out, exist_ok=True)
os.makedirs(val_out, exist_ok=True)

# Loop through each emotion folder
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    
    if not os.path.isdir(class_path):
        continue
    
    images = os.listdir(class_path)
    random.shuffle(images)
    
    split_index = int(len(images) * split_ratio)
    
    train_imgs = images[:split_index]
    val_imgs = images[split_index:]
    
    # Create class folders in output
    os.makedirs(os.path.join(train_out, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_out, class_name), exist_ok=True)
    
    # Copy train images
    for img in train_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_out, class_name, img)
        shutil.copy(src, dst)
    
    # Copy val images
    for img in val_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_out, class_name, img)
        shutil.copy(src, dst)

print("✅ Dataset split complete!")