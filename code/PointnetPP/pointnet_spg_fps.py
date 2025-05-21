import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, classification_report


# -------------------------------
# Datenladen und Sampling-Utilities
# -------------------------------

def farthest_point_sampling(xyz, npoint):
    """
    Farthest Point Sampling (FPS)
    Eingabe:
        xyz: (N, 3) Array mit Punktkoordinaten
        npoint: Anzahl der zu samplenden Punkte
    Rückgabe:
        centroids: (npoint,) Array der Indizes der ausgewählten Punkte
    """
    # Falls torch.Tensor, nach NumPy konvertieren
    if isinstance(xyz, torch.Tensor):
        pts = xyz.cpu().numpy()
    else:
        pts = xyz
    N, _ = pts.shape
    centroids = np.zeros(npoint, dtype=np.int64)
    distance = np.ones(N) * 1e10
    # Zufälligen Startpunkt wählen
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = pts[farthest, :3]
        # Quadratische Distanz zu letztem Zentrumspunkt berechnen
        dist = np.sum((pts[:, :3] - centroid) ** 2, axis=1)
        # Nächste Distanz aktualisieren
        mask = dist < distance
        distance[mask] = dist[mask]
        # Neuen weitesten Punkt wählen
        farthest = np.argmax(distance)
    return centroids

class LiDARPointCloudDataset(Dataset):
    """
    Dataset für LiDAR-Punktwolken-Segmentierung.
    Erwartet Punkt- und Label-.npy-Dateien mit mindestens num_points.
    Integriert FPS- und instanz-basiertes Sampling.
    """
    def __init__(self, data_dir, pattern_points="*_points.npy", pattern_labels="*_labels.npy", num_points=4096, ignore_label=7):
        super().__init__()
        # Dateipfade sammeln
        self.point_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("_points.npy")])
        self.label_files = [pf.replace("_points.npy", "_labels.npy") for pf in self.point_files]
        self.num_points = num_points
        self.ignore_label = ignore_label
        all_labels = [] #-------------------------
        for f in self.label_files:
            raw = np.load(f)
            all_labels.extend(raw.tolist())
        uniques = sorted(set(all_labels) - {ignore_label}) #-------------------------
        self.class_map = {orig: i for i, orig in enumerate(uniques)}
        self.num_classes = len(uniques) #-------------------------


    def __len__(self):
        return len(self.point_files)

    def __getitem__(self, idx):
        # Punktwolke und Labels laden
        points = np.load(self.point_files[idx])  # Form: (N, 3 [+ Features])
        labels = np.load(self.label_files[idx]).astype(int)  # Rohlabels

        # 2) Mappen auf 0…C-1 (ignore_label bleibt außerhalb)
        mapped = np.full_like(labels, fill_value=self.ignore_label, dtype=int)
        for orig, new in self.class_map.items():
            mapped[labels == orig] = new
        labels = mapped  # Jetzt alle Werte ∈ [0…num_classes-1] ∪ {ignore_label}

        N = points.shape[0]
        # Wenn mehr Punkte als num_points, anwenden von FPS
        if N > self.num_points:
            inds = farthest_point_sampling(points[:, :3], self.num_points)
            points = points[inds, :]
            labels = labels[inds]
        # Wenn weniger Punkte, auffüllen (hier per Wiederholung)
        elif N < self.num_points:
            dup_inds = np.random.choice(N, self.num_points - N, replace=True)
            points = np.concatenate([points, points[dup_inds, :]], axis=0)
            labels = np.concatenate([labels, labels[dup_inds]], axis=0)

        # Instanz-basiertes Sampling:
        # Sicherstellen, dass jede Klasse im Sample mindestens einmal vorkommt
        unique_cls = np.unique(labels)
        for cls in unique_cls:
            if cls == self.ignore_label:
                continue
            if np.sum(labels == cls) == 0:
                # Falls Klasse im Original vorhanden, fügen wir einen Punkt hinzu
                orig_class_inds = np.where(labels == cls)[0]
                if len(orig_class_inds) == 0:
                    continue
                add_idx = np.random.choice(orig_class_inds)
                # Ersetze Punkt aus der momentan größten Klasse
                class_counts = {c: np.sum(labels == c) for c in unique_cls if c != self.ignore_label}
                major = max(class_counts, key=class_counts.get)
                major_inds = np.where(labels == major)[0]
                remove_idx = np.random.choice(major_inds) if len(major_inds) > 1 else np.random.randint(0, self.num_points)
                points[remove_idx] = points[add_idx]
                labels[remove_idx] = labels[add_idx]

        # In Torch-Tensor konvertieren
        points = torch.from_numpy(points).float().contiguous()  # (num_points, D)
        labels = torch.from_numpy(labels).long()   # (num_points,)

        return points, labels

# -------------------------------
# Lightning-Modul für Training
# -------------------------------

class PointNet2Segmentation(pl.LightningModule):
    """
    PyTorch Lightning Modul für PointNet++ Semantische Segmentierung mit SPG-Loss.
    """
    def __init__(self, model, num_classes=13, ignore_label=7, spg_weight=0.1, lr=1e-3):
        """
        Args:
            model: PointNet2-Segmentierungsmodell (z.B. PointNet2SemSegSSG)
            num_classes: Anzahl der Klassen (inklusive Ignorierklasse)
            ignore_label: Label, das im Loss ignoriert wird
            spg_weight: Gewichtungsfaktor für den SPG-Loss
            lr: Lernrate
        """
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        self.spg_weight = spg_weight
        self.lr = lr

    def forward(self, x):
        # Vorwärtsdurchlauf durch das Modell
        # Erwartet Eingabe (B, N, 3 [+ Features])
        logits = self.model(x)  # Ausgabe: (B, num_classes, N)
        return logits

    def training_step(self, batch, batch_idx):
        points, labels = batch  # points: (B, N, D), labels: (B, N)
        points = points.to(self.device).contiguous()
        logits = self(points)  # (B, C, N)
        labels = labels.to(logits.device)

        # Cross-Entropy-Loss (ignoriert die Ausreißer-Klasse)
        ce_loss = self.ce_loss(logits, labels)

        # Subspace Prototype Guidance (SPG) Loss
        # Merkmale vor letzter Klassifikationsschicht extrahieren
        xyz = points[..., :3]  # (B, N, 3)
        if points.size(-1) > 3:
            feats_input = points[..., 3:].permute(0, 2, 1).contiguous()  # (B, C_in, N)
        else:
            feats_input = None

        l_xyz = [xyz]
        l_features = [feats_input]
        # Set-Abstraction-Module anwenden
        for sa in self.model.SA_modules:
            cur_xyz   = l_xyz[-1].contiguous()                  
            cur_feats = l_features[-1]
            if cur_feats is not None:
                cur_feats = cur_feats.contiguous()               
            li_xyz, li_feats = sa(cur_xyz, cur_feats)
            l_xyz.append(li_xyz.contiguous())                  
            l_features.append(li_feats.contiguous())              

        # FP‑Module rückwärts
        for i in range(len(self.model.FP_modules)):
            idx = -1 - i
            before = l_xyz[idx-1].contiguous()                 
            after  = l_xyz[idx].contiguous()                    
            f1     = l_features[idx-1].contiguous()                
            f2     = l_features[idx].contiguous()                  
            l_features[idx-1] = self.model.FP_modules[idx](before, after, f1, f2).contiguous()
        # l_features[0] hat Form (B, 128, N)
        point_feats = l_features[0]  # (B, C_feat, N)

        # Auf FC-Layers anwenden (bis vorletzter Conv1d)
        fc_layers = list(self.model.fc_lyaer.children())
        feat_extractor = nn.Sequential(*fc_layers[:-1])
        feats = feat_extractor(point_feats)  # (B, C_feat, N)
        B, C_feat, N = feats.shape
        feats = feats.permute(0, 2, 1).reshape(-1, C_feat)  # (B*N, C_feat)
        labels_flat = labels.view(-1)  # (B*N,)

        # Klassen-Prototypen berechnen und SPG-Loss
        spg_loss = torch.tensor(0.0, device=logits.device)
        for cls in range(self.num_classes):
            if cls == self.ignore_label:
                continue
            mask = labels_flat == cls
            if torch.any(mask):
                cls_feats = feats[mask]
                center = cls_feats.mean(dim=0, keepdim=True)  # (1, C_feat)
                spg_loss = spg_loss + ((cls_feats - center).pow(2).sum()) / (cls_feats.shape[0] + 1e-6)
        # SPG-Loss normalisieren
        spg_loss = spg_loss / (self.num_classes - 1)

        total_loss = ce_loss + self.spg_weight * spg_loss


        return {
            "loss":            total_loss,
            "train_ce_loss":   ce_loss.detach().item(),
            "train_spg_loss":  spg_loss.detach().item(),
            "train_total_loss": total_loss.detach().item()
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        points, labels = batch
        logits = self.forward(points)             # (B, C, N)
        loss = self.ce_loss(logits, labels)
        preds = logits.argmax(dim=1)
        acc   = (preds == labels).float().mean()
        # Rückgabe als Dict: Lightning sammelt diese automatisch
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        # outputs: Liste der zurückgegebenen Dicts der validation_step
        avg_loss = sum(o["val_loss"] for o in outputs) / len(outputs)
        avg_acc  = sum(o["val_acc"] for o in outputs) / len(outputs)

        return {
                "val_loss": avg_loss,
                "val_acc": avg_acc
                 }
    
    def test_dataloader(self):
        test_ds = LiDARPointCloudDataset(data_dir=TEST_DIR, num_points=NUM_POINTS, ignore_label=IGNORE_LABEL)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)
        return test_loader

    def test_step(self, batch, batch_idx):
        points, labels = batch
        logits = self.forward(points)                  # (B, C, N)
        preds  = logits.argmax(dim=1)                  # (B, N)
        # Gibt NumPy‑Arrays zurück, damit wir im Endhook leichter aggregieren:
        return {
            "preds": preds.detach().cpu(),
            "labels": labels.detach().cpu()
        }

    def test_epoch_end(self, outputs):
        # outputs = Liste von Dicts aus test_step
        all_preds  = torch.cat([x["preds"]  for x in outputs], dim=0).numpy().ravel()
        all_labels = torch.cat([x["labels"] for x in outputs], dim=0).numpy().ravel()

        # Accuracy
        acc = accuracy_score(all_labels, all_preds)
        # Per‑Class F1
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        # Vollständiger Report
        report = classification_report(all_labels,
                                       all_preds,
                                       zero_division=0,
                                       digits=4)

        # Ausgabe
        print("\n===== Test Results =====")
        print(f"Overall Accuracy: {acc:.4f}")
        print("F1 per class:", f1_per_class)
        print("\nClassification Report:\n", report)

        # optional: ins Lightning‑Logger schreiben
        metrics = {"test_acc": acc}
        for i, f1c in enumerate(f1_per_class):
            metrics[f"test_f1_class_{i}"] = f1c
        self.logger.log_metrics(metrics, step=self.global_step)
        return metrics

# -------------------------------
# DataLoader und Trainer vorbereiten
# -------------------------------

if __name__ == "__main__":
    # Parameter
    DATA_DIR = '/cfs/earth/scratch/nogernic/Pointnet2_PyTorch/train'# Verzeichnis mit *_points.npy und *_labels.npy
    VAL_DIR =  '/cfs/earth/scratch/nogernic/Pointnet2_PyTorch/val'
    TEST_DIR = '/cfs/earth/scratch/nogernic/Pointnet2_PyTorch/test'
    BATCH_SIZE = 8
    NUM_POINTS = 4096
    IGNORE_LABEL = 7
    NUM_CLASSES = 13  # inklusive Ausreißer-Klasse falls vorhanden
    LR = 1e-3
    SPG_WEIGHT = 0.1
    MAX_EPOCHS = 10
    MODEL_SAVE_PATH = '/cfs/earth/scratch/nogernic/Pointnet2_PyTorch/models/pointnet2_ssg_sem.pth'
    INPUT_CHANNELS = 3 #--------------------------


    # Dataset erstellen
    train_ds = LiDARPointCloudDataset(data_dir=DATA_DIR, num_points=NUM_POINTS, ignore_label=IGNORE_LABEL)
    val_ds = LiDARPointCloudDataset(data_dir=VAL_DIR, num_points=NUM_POINTS, ignore_label=IGNORE_LABEL)


    # (Optional) WeightedRandomSampler zur Klassenbalancierung
    sample_weights = [1.0] * len(train_ds)  # hier gleichverteilt (anpassen nach Bedarf)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=1, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)


    NUM_CLASSES = train_ds.num_classes #--------------------------
    hparams = {
        'num_points': NUM_POINTS,
        'num_classes': NUM_CLASSES,
        'input_channels': INPUT_CHANNELS,
        'model.use_xyz': True,
    }
    # PointNet2-Modell (Erik Wijmans Implementierung)
    # Erfordert Installation und Auffindbarkeit der Bibliothek
    from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG
    model = PointNet2SemSegSSG(hparams) #--------------------------

    # Lightning-Modul instanziieren
    lit_model = PointNet2Segmentation(model=model, num_classes=NUM_CLASSES,
                                      ignore_label=IGNORE_LABEL,
                                      spg_weight=SPG_WEIGHT, lr=LR)

    # Trainer
    trainer = pl.Trainer(gpus=2 if torch.cuda.is_available() else 0, max_epochs=MAX_EPOCHS, log_every_n_steps=10)
    trainer.fit(lit_model, train_loader, val_loader)

    trainer.test(lit_model)

        # Modell speichern
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print('Modell gespeichert unter:', MODEL_SAVE_PATH)
