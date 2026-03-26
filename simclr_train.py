import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ======================
# CUDA SETTINGS
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.backends.cudnn.benchmark = True  # speeds up training

# ======================
# DATASET LOADER
# ======================
class UnlabeledDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.images = os.listdir(folder)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.images[idx])
        img = Image.open(path).convert("RGB")

        img1 = self.transform(img)
        img2 = self.transform(img)

        return img1, img2

# ======================
# AUGMENTATIONS
# ======================
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
])

dataset = UnlabeledDataset(r"D:\CLOUD PROJ\fer2013\ssl_data\all_images", transform)

loader = DataLoader(
    dataset,
    batch_size=128,          # larger batch for GPU
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# ======================
# SIMCLR MODEL
# ======================
class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z

model = SimCLR().to(device)

# ======================
# CONTRASTIVE LOSS
# ======================
def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.shape[0]

    z = torch.cat([z1, z2], dim=0)
    z = nn.functional.normalize(z, dim=1)

    similarity_matrix = torch.matmul(z, z.T)

    # create mask to remove self similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)

    # positive pairs
    positives = torch.cat([
        torch.diag(similarity_matrix, batch_size),
        torch.diag(similarity_matrix, -batch_size)
    ]).view(2 * batch_size, 1)

    # negatives
    negatives = similarity_matrix[~mask].view(2 * batch_size, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits = logits / temperature

    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(device)

    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

# ======================
# TRAINING SETUP
# ======================
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
epochs = 50

# ======================
# TRAINING LOOP
# ======================
for epoch in range(epochs):
    model.train()
    total_loss = 0

    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

    for x1, x2 in loop:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        z1 = model(x1)
        z2 = model(x2)

        loss = nt_xent_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

# ======================
# SAVE ENCODER
# ======================
torch.save(model.encoder.state_dict(), "simclr_encoder.pth")

print("\n✅ Pretraining Complete — Encoder Saved!")