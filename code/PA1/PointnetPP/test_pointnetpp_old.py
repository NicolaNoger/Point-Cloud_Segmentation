import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
# Korrektes Semantic-Segmentation-Modul importieren
from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG

import inspect
print(">>> Geladene Modell-Quelle:", inspect.getsourcefile(PointNet2SemSegSSG))


# Konfiguration
INPUT_CHANNELS = 3    # Anzahl zusätzlicher Features (intensity, return_number, number_of_returns)
NUM_CLASSES = 10      # Anzahl der Segmentklassen
DATA_DIR = '/cfs/earth/scratch/nogernic/Pointnet2_PyTorch/data'
BATCH_SIZE = 4        # Anzahl Dateien/Samples pro Batch
NUM_WORKERS = 2       # Prozesse zum Laden der Daten
EPOCHS = 5
LR = 0.001
MODEL_SAVE_PATH = '/cfs/earth/scratch/nogernic/Pointnet2_PyTorch/models/pointnet2_ssg_sem.pth'

class NPYPointCloudDataset(Dataset):
    """
    Lädt Dateien:
      - tileCH_chunkX_points.npy
      - tileCH_chunkX_labels.npy
    im selben Ordner.
    """
    def __init__(self, data_dir, transform=None):
        self.label_files = sorted(glob.glob(os.path.join(data_dir, '*_labels.npy')))
        self.transform = transform

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        label_file = self.label_files[idx]
        point_file = label_file.replace('_labels.npy', '_points.npy')

        # Laden
        points = np.load(point_file)    # Shape [N, 6]
        labels = np.load(label_file)    # Shape [N]

        # XYZ normalisieren (Mittelpunkt auf 0, Einheitsskala)
        coords = points[:, :3]
        coords = coords - coords.mean(axis=0)
        coords = coords / (np.max(np.linalg.norm(coords, axis=1)) + 1e-6)

        # Zusätzliche Features intensity, return_number, number_of_returns
        features = points[:, 3:].astype(np.float32)

        # In Tensor-Form bringen: [N, C]
        inp = np.concatenate([coords, features], axis=1)  # [N, 6]
        inp = torch.tensor(inp, dtype=torch.float32)      # [N, C]
        labels = torch.tensor(labels, dtype=torch.long)    # [N]

        return inp, labels

if __name__ == '__main__':
    # Sicherstellen, dass Modelle-Ordner existiert
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Dataset & DataLoader
    dataset = NPYPointCloudDataset(DATA_DIR)
    # Note: Modell erwartet Eingabe [B, N, C]
    # DataLoader liefert (inputs: [B, N, C], labels: [B, N])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS)

    # Gerät
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Nutze Gerät: {device} | Batch size: {BATCH_SIZE} Samples pro Batch')

    hparams = {
    "num_points": 4096,
    "num_classes": 9,        
    "input_channels": 3,      
    "model.use_xyz": True,    
    }
    # Modell
    model = PointNet2SemSegSSG(hparams)
    print(">>> Modell erwartet in_channels:", model.SA_modules[0].mlps[0][0].in_channels)
    model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for inputs, targets in loader:
            # inputs: [B, N, C], targets: [B, N]
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)      # [B, num_classes, N]
            # CrossEntropyLoss erwartet [B, C, N] und [B, N]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch+1}/{EPOCHS} – Loss: {avg_loss:.4f}')

    # Modell speichern
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'Modell gespeichert unter: {MODEL_SAVE_PATH}')
