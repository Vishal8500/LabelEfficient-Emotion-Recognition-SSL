import os
import shutil

# ===== CHANGE PATHS IF NEEDED =====
train_dir = "fer2013/train"
val_dir = "fer2013/val"
ssl_output = "fer2013/ssl_data/all_images"
# ==================================

os.makedirs(ssl_output, exist_ok=True)

def copy_all_images(source_dir):
    count = 0
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        for img in os.listdir(class_path):
            src = os.path.join(class_path, img)
            dst = os.path.join(ssl_output, f"{count}_{img}")
            
            shutil.copy(src, dst)
            count += 1
            
    return count

total = 0
total += copy_all_images(train_dir)
total += copy_all_images(val_dir)

print(f"✅ Unlabeled dataset created with {total} images!")