import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd
import os
import numpy as np

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
    "baseline_20": "baseline_train_20.pth",
    "baseline_40": "baseline_train_40.pth",
    "baseline_80": "baseline_train_80.pth",
    "baseline_100": "baseline_resnet18_emotion100.pth",
    "simclr_20": "simclr_train_20.pth",
    "simclr_40": "simclr_train_40.pth",
    "simclr_80": "simclr_train_80.pth",
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

# ---------------- CONFUSION MATRIX FUNCTIONS ----------------
def save_confusion_matrix(cm, name):
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 10}
    )

    plt.title(f"Confusion Matrix: {name}", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(CM_DIR, f"{name}_cm.png"), dpi=300)
    plt.close()


def save_normalized_cm(cm, name):
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f"Normalized Confusion Matrix: {name}", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)

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

    print(f"\n{name} → Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

    # ---------- Classification Report ----------
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(REPORT_DIR, f"{name}_report.csv"))

    # ---------- Confusion Matrix ----------
    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, name)
    save_normalized_cm(cm, name)

    return acc, f1, cm, report_df

# ---------------- RUN ALL ----------------
results = []
all_cm = {}
all_reports = {}

for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        model = load_model(path)
        acc, f1, cm, report = evaluate(model, name)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Macro_F1": f1
        })

        all_cm[name] = cm
        all_reports[name] = report
    else:
        print(f"{path} not found!")

df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, "final_results.csv"), index=False)

print("\n===== FINAL RESULTS =====")
print(df)

# ---------------- PLOTS ----------------

# Sort for consistency
df = df.sort_values("Model")

# 1️⃣ Accuracy Plot
plt.figure()
for model_type in ["baseline", "simclr"]:
    subset = df[df["Model"].str.contains(model_type)]
    x = [20,40,80,100]
    plt.plot(x, subset["Accuracy"], marker='o', label=model_type)

plt.xlabel("Label Fraction (%)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Label Fraction")
plt.legend()
plt.grid()
plt.savefig(os.path.join(PLOT_DIR, "accuracy_plot.png"), dpi=300)
plt.close()

# 2️⃣ F1 Plot
plt.figure()
for model_type in ["baseline", "simclr"]:
    subset = df[df["Model"].str.contains(model_type)]
    x = [20,40,80,100]
    plt.plot(x, subset["Macro_F1"], marker='o', label=model_type)

plt.xlabel("Label Fraction (%)")
plt.ylabel("Macro F1")
plt.title("Macro F1 vs Label Fraction")
plt.legend()
plt.grid()
plt.savefig(os.path.join(PLOT_DIR, "f1_plot.png"), dpi=300)
plt.close()

# 3️⃣ Bar Plot
df.set_index("Model")[["Accuracy", "Macro_F1"]].plot(kind="bar")
plt.title("Model Comparison")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "bar_plot.png"), dpi=300)
plt.close()

# 4️⃣ Per-Class F1 (best model)
best_model = df.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
best_report = all_reports[best_model]

f1_scores = best_report.loc[class_names]["f1-score"]

plt.figure()
f1_scores.plot(kind='bar')
plt.title(f"Per-Class F1 ({best_model})")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "per_class_f1.png"), dpi=300)
plt.close()

# 5️⃣ Confusion Matrix Comparison
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.heatmap(all_cm["baseline_100"], cmap="Blues")
plt.title("Baseline 100")

plt.subplot(1,2,2)
sns.heatmap(all_cm["simclr_100"], cmap="Blues")
plt.title("SimCLR 100")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "cm_comparison.png"), dpi=300)
plt.close()

# 6️⃣ SSL Gain Plot
baseline_acc = df[df["Model"].str.contains("baseline")]["Accuracy"].values
ssl_acc = df[df["Model"].str.contains("simclr")]["Accuracy"].values

gain = ssl_acc - baseline_acc

plt.figure()
plt.plot([20,40,80,100], gain, marker='o')
plt.title("SSL Gain over Baseline")
plt.xlabel("Label Fraction")
plt.ylabel("Accuracy Gain")
plt.grid()
plt.savefig(os.path.join(PLOT_DIR, "ssl_gain.png"), dpi=300)
plt.close()

print("\n✅ All outputs saved in 'outputs/' folder")