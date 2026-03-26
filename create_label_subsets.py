import os
import random
import shutil

random.seed(42)

source_dir = "fer2013/train"
output_base = "fer2013"

splits = {
    "train_80": 0.8,
    "train_40": 0.4,
    "train_20": 0.2
}

for split_name, ratio in splits.items():
    output_dir = os.path.join(output_base, split_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        num_keep = int(len(images) * ratio)
        selected = images[:num_keep]

        out_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(out_class_dir, exist_ok=True)

        for img in selected:
            src = os.path.join(class_path, img)
            dst = os.path.join(out_class_dir, img)
            shutil.copy(src, dst)

    print(f"✅ Created {split_name} dataset")