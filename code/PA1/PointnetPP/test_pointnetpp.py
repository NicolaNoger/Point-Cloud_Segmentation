import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG
from pytorch_lightning import Trainer


# --- Konfiguration ---
NUM_POINTS = 4096                      # feste Punkte pro Sample
INPUT_CHANNELS = 3                     # Zusatzfeatures: intensity, return_number, number_of_returns
DATA_DIR = '/cfs/earth/scratch/nogernic/Pointnet2_PyTorch/data'
BATCH_SIZE = 4                         # Anzahl Samples pro Batch
NUM_WORKERS = 1                        # Prozesse zum Laden (HPC empfiehlt 1)
EPOCHS = 5
LR = 0.001
MODEL_SAVE_PATH = '/cfs/earth/scratch/nogernic/Pointnet2_PyTorch/models/pointnet2_ssg_sem.pth'

class NPYPointCloudDataset(Dataset):
    """
    Dataset für .npy-Batches mit Punkten und Labels.
    Erstellt ein Mapping der raw-Labels auf 0…K-1 und sampelt jede Wolke auf NUM_POINTS.
    """
    def __init__(self, data_dir, num_points, transform=None):
        self.label_files = sorted(glob.glob(os.path.join(data_dir, '*_labels.npy')))
        self.num_points = num_points
        self.transform = transform
        # Klassen-Mapping vorbereiten
        all_labels = []
        for f in self.label_files:
            raw = np.load(f)
            all_labels.extend(raw.tolist())
        uniques = sorted(set(all_labels))
        self.class_map = {v: i for i, v in enumerate(uniques)}
        self.num_classes = len(uniques)

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        label_file = self.label_files[idx]
        point_file = label_file.replace('_labels.npy', '_points.npy')
        points = np.load(point_file)  # [N, 6]
        raw_labels = np.load(label_file)  # [N]

        # Sampling auf fixierte Größe
        N = points.shape[0]
        if N >= self.num_points:
            idxs = np.random.choice(N, self.num_points, replace=False)
        else:
            idxs = np.random.choice(N, self.num_points, replace=True)
        points = points[idxs]
        raw_labels = raw_labels[idxs]

        # Label remapping: raw → 0…(K-1)
        labels = np.array([self.class_map[v] for v in raw_labels], dtype=np.int64)

        # XYZ normalisieren
        coords = points[:, :3]
        coords = coords - coords.mean(axis=0)
        coords = coords / (np.max(np.linalg.norm(coords, axis=1)) + 1e-6)

        # Zusatzfeatures
        features = points[:, 3:].astype(np.float32)

        # Input tensor: [N, C]
        inp = np.concatenate([coords, features], axis=1)
        inp = torch.tensor(inp, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # Debug Label-Bereich
        print(f"Labels min/max für Sample {idx}: {labels.min().item()} / {labels.max().item()}")
        assert labels.min() >= 0 and labels.max() < self.num_classes, \
            f"Labels außerhalb [0,{self.num_classes-1}]: {labels.min().item()}…{labels.max().item()}"

        return inp, labels

if __name__ == '__main__':
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Dataset & DataLoader
    dataset = NPYPointCloudDataset(DATA_DIR, num_points=NUM_POINTS)
    loader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=NUM_WORKERS)

    # Dynamische Klassenanzahl
    NUM_CLASSES = dataset.num_classes
    print(f"Verwendete Klassen: {NUM_CLASSES}")

    # Gerät
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Gerät: {device} | Batch size: {BATCH_SIZE} | Num_workers: {NUM_WORKERS}')

    # Hyperparameter für Modell
    hparams = {
        'num_points': NUM_POINTS,
        'num_classes': NUM_CLASSES,
        'input_channels': INPUT_CHANNELS,
        'model.use_xyz': True,
    }

    # Modell laden und prüfen
    model = PointNet2SemSegSSG(hparams).to(device)
    in_ch = model.SA_modules[0].mlps[0][0].in_channels
    print('>>> SA[0] erwartet in_channels:', in_ch)  # sollte 6 sein

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    trainer = Trainer(
    accelerator="gpu",
    devices=4,
    strategy="ddp",
    max_epochs=100
)

    # Training
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  # [B, num_classes, N]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.4f}')

    # Modell speichern
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print('Modell gespeichert unter:', MODEL_SAVE_PATH)
