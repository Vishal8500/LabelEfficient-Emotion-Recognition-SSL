import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
BATCH_SIZE = 64

TEST_DIR = r"D:\CLOUD PROJ\fer2013\test"

OUTPUT_DIR = "outputs"
CM_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

MODEL_PATHS = {
    "baseline_100": "baseline_resnet18_emotion100.pth",
    "simclr_100": "simclr_finetuned100_emotion.pth"
}

# ---------------- DATA ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = test_dataset.classes

# ---------------- MODEL ----------------
def load_model(path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ---------------- CONFUSION MATRIX ----------------
def save_confusion_matrix(cm, name):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(CM_DIR, f"{name}_cm.png"), dpi=300)
    plt.close()

def save_normalized_cm(cm, name):
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8,6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f"Normalized CM: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(CM_DIR, f"{name}_cm_norm.png"), dpi=300)
    plt.close()

# ---------------- EVALUATION ----------------
def evaluate(model, name):
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"{name} → Acc: {acc:.4f}, F1: {f1:.4f}")

    # report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(REPORT_DIR, f"{name}_report.csv"))

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, name)
    save_normalized_cm(cm, name)

    return acc, f1, cm, report_df

# ---------------- TSNE ----------------
def extract_features(model):
    features, labels = [], []

    with torch.no_grad():
        for images, lbls in test_loader:
            images = images.to(DEVICE)

            x = model.conv1(images)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)

            x = model.avgpool(x)
            x = torch.flatten(x, 1)

            features.append(x.cpu().numpy())
            labels.extend(lbls.numpy())

    return np.concatenate(features), np.array(labels)

def plot_tsne(model, name):
    features, labels = extract_features(model)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(7,6))
    for i, cls in enumerate(class_names):
        idx = labels == i
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=cls, s=10)

    plt.legend()
    plt.title(f"t-SNE: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_tsne.png"), dpi=300)
    plt.close()

# ---------------- PER CLASS ----------------
def plot_per_class(all_reports):
    b = all_reports["baseline_100"].loc[class_names]["f1-score"]
    s = all_reports["simclr_100"].loc[class_names]["f1-score"]

    x = np.arange(len(class_names))

    plt.figure()
    plt.bar(x-0.2, b, 0.4, label="Baseline")
    plt.bar(x+0.2, s, 0.4, label="SimCLR")

    plt.xticks(x, class_names, rotation=45)
    plt.title("Per-Class F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "per_class.png"), dpi=300)
    plt.close()

# ---------------- MISCLASSIFIED ----------------
def plot_errors(model, name):
    imgs, t, p = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(images)):
                if preds[i] != labels[i]:
                    imgs.append(images[i].cpu())
                    t.append(labels[i].item())
                    p.append(preds[i].item())
                if len(imgs) >= 12:
                    break
            if len(imgs) >= 12:
                break

    plt.figure(figsize=(10,8))
    for i in range(len(imgs)):
        plt.subplot(3,4,i+1)
        plt.imshow(imgs[i].permute(1,2,0))
        plt.title(f"T:{class_names[t[i]]}\nP:{class_names[p[i]]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_errors.png"), dpi=300)
    plt.close()

# ---------------- MAIN ----------------
results = []
all_reports = {}

for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        model = load_model(path)
        acc, f1, cm, rep = evaluate(model, name)

        results.append({"Model": name, "Accuracy": acc, "Macro_F1": f1})
        all_reports[name] = rep

        plot_tsne(model, name)
        plot_errors(model, name)

df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)

plot_per_class(all_reports)

print("✅ DONE — All outputs saved in 'outputs/' folder")