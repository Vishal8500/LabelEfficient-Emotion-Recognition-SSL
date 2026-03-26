import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# CONFIG
# ======================
label_splits = ["train_80", "train_40", "train_20"]
epochs = 20
batch_size = 64

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder("fer2013/test", transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

num_classes = len(test_dataset.classes)

# ======================
# TRAIN FUNCTION
# ======================
def train_model(train_path, use_simclr, split_name):

    train_dataset = datasets.ImageFolder(train_path, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = resnet18(weights=None)
    model.fc = nn.Linear(512, num_classes)

    if use_simclr:
        encoder_weights = torch.load("simclr_encoder.pth")
        model.load_state_dict(encoder_weights, strict=False)
        print("Loaded SimCLR pretrained encoder")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ================= TRAIN =================
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"{split_name} {'SimCLR' if use_simclr else 'Baseline'} Epoch {epoch+1}")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}")

    # ================= SAVE MODEL =================
    model_type = "simclr" if use_simclr else "baseline"
    save_name = f"{model_type}_{split_name}.pth"
    torch.save(model.state_dict(), save_name)
    print(f"Saved: {save_name}")

    # ================= EVALUATION =================
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
    print("="*60)


# ======================
# RUN ALL EXPERIMENTS
# ======================
for split in label_splits:
    train_path = f"fer2013/{split}"

    print("\n" + "#"*70)
    print(f"RUNNING BASELINE FOR {split}")
    print("#"*70)

    train_model(train_path, use_simclr=False, split_name=split)

    print("\n" + "#"*70)
    print(f"RUNNING SIMCLR FINE-TUNING FOR {split}")
    print("#"*70)

    train_model(train_path, use_simclr=True, split_name=split)

print("\nALL EXPERIMENTS COMPLETED!")