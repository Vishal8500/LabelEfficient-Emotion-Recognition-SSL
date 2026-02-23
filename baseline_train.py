import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# DATA TRANSFORMS
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ======================
# LOAD DATASETS
# ======================
train_dataset = datasets.ImageFolder("fer2013/train", transform)
val_dataset = datasets.ImageFolder("fer2013/val", transform)
test_dataset = datasets.ImageFolder("fer2013/test", transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=0)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)

# ======================
# BASELINE MODEL (NO SSL)
# ======================
model = resnet18(weights=None)   # RANDOM INITIALIZATION
model.fc = nn.Linear(512, num_classes)

model = model.to(device)

# ======================
# TRAINING SETUP
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 20

# ======================
# TRAINING LOOP
# ======================
for epoch in range(epochs):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

print("✅ Baseline training complete!")

# ======================
# SAVE BASELINE MODEL
# ======================
torch.save(model.state_dict(), "baseline_resnet18_emotion.pth")
print("💾 Baseline model saved!")

# ======================
# EVALUATION
# ======================
model.eval()

all_preds = []
all_labels = []

print("\nEvaluating on test set...")

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ======================
# CLASSIFICATION REPORT
# ======================
print("\n📊 Baseline Classification Report:\n")

print(classification_report(
    all_labels,
    all_preds,
    target_names=train_dataset.classes
))

# ======================
# CONFUSION MATRIX
# ======================
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:\n", cm)