# === Imports ===
import os
import glob
import pandas as pd
import numpy as np
import laspy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import defaultdict


# Check for GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs found: {gpus}")
    try:
        # Enable memory growth to prevent TensorFlow from grabbing all memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found. Training will run on CPU.")


# --------------------------------------------------------------------------
# === Data preparation ===
# --------------------------------------------------------------------------
# === Automatically find all .laz files ===
folder_path = os.path.abspath("/cfs/earth/scratch/klaveflo/PA1/data/training")
file_paths = glob.glob(os.path.join(folder_path, "*.laz"))

if not file_paths:
    raise FileNotFoundError(f"No .laz files found in {folder_path}")
else:
    print(f"Found {len(file_paths)} LAZ files.")


# === Prepare containers ===
all_features = []
all_labels = []

# === Create label map based on all files together ===
# (First pass to get all unique labels)
all_labels_raw = []
for path in file_paths:
    las_temp = laspy.read(path)
    all_labels_raw.extend(np.unique(las_temp.classification))

unique_labels = np.unique(all_labels_raw)
# Remove class 15
unique_labels = unique_labels[unique_labels != 15]

label_map = {label: idx for idx, label in enumerate(unique_labels)}

# === load and process all files ===
for path in file_paths:
    las = laspy.read(path)


    # Extract features
    ground_height = np.min(las.z)
    X = np.vstack((
        las.x,
        las.y,
        las.z,
        las.intensity,
        las.return_number,
        las.number_of_returns,
        las.z - ground_height,
        las.scan_angle
    )).T.astype(np.float32)

    y_raw = np.array(las.classification)

    # Remove points with class 15 (error class)
    mask_valid = (y_raw != 15)
    X = X[mask_valid]
    y_raw = y_raw[mask_valid]

    # map labels
    y_mapped = np.array([label_map[label] for label in y_raw])


    all_features.append(X)
    all_labels.append(y_mapped)

# === Stack everything together ===
X_combined = np.vstack(all_features)
y_combined = np.hstack(all_labels)

# Class distribution
class_counts = pd.Series(y_combined).value_counts().sort_index()


# --------------------------------------------------------------------------
# === Preprocessing ===
# --------------------------------------------------------------------------

# === Standardize features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# === Skip SMOTE for now ===
X_resampled, y_resampled = X_scaled, y_combined

# === Center each point cloud ===
N_POINTS = 1024
X_centered = X_resampled.copy()

for i in range(0, len(X_centered), N_POINTS):
    chunk = X_centered[i:i+N_POINTS]
    centroid = np.mean(chunk[:, :3], axis=0)
    X_centered[i:i+N_POINTS, :3] -= centroid

# Parameters
target_samples_per_class = 5_000  # adjusted based on class with least samples

# Group points by class
class_indices = defaultdict(list)
for idx, label in enumerate(y_resampled):
    class_indices[label].append(idx)

# Balanced indices: keep all rare, limit common
balanced_indices = []
for label, indices in class_indices.items():
    if len(indices) <= target_samples_per_class:
        balanced_indices.extend(indices)
    else:
        balanced_indices.extend(np.random.choice(indices, target_samples_per_class, replace=False))

# Shuffle indices to mix
np.random.shuffle(balanced_indices)

# Get balanced X and y
X_balanced = X_centered[balanced_indices]
y_balanced = y_resampled[balanced_indices]

# Trim to fit full point clouds
num_samples = len(X_balanced) // N_POINTS
X_balanced = X_balanced[:num_samples * N_POINTS]
y_balanced = y_balanced[:num_samples * N_POINTS]

# Reshape into (num_clouds, 1024, features)
X_pointcloud = X_balanced.reshape(num_samples, N_POINTS, -1)
y_pointcloud = y_balanced.reshape(num_samples, N_POINTS)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pointcloud, y_pointcloud, test_size=0.2, random_state=42)



# ---------------------------------------------------------------------------
# === PointNet Model ===
# ---------------------------------------------------------------------------

def build_pointnet_model(num_points, num_features, num_classes):
    inputs = tf.keras.Input(shape=(num_points, num_features))

    # Per-point MLP (shared weights)
    x = layers.Conv1D(64, 1, activation='relu')(inputs)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.Conv1D(1024, 1, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Global feature (max pooling across points)
    global_feat = layers.GlobalMaxPooling1D()(x)
    
    # Repeat global features for each point
    global_feat = layers.RepeatVector(num_points)(global_feat)
    global_feat = layers.Concatenate()([x, global_feat])  # concat local + global

    # Bigger shared MLP
    x = layers.Conv1D(512, 1, activation='relu')(global_feat)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(512, 1, activation='relu')(x)   
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(256, 1, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, 1, activation='relu')(x)     
    x = layers.Dropout(0.3)(x) 

    # Output layer
    x = layers.Conv1D(num_classes, 1, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=x)
    return model

NUM_CLASSES = len(np.unique(y_pointcloud))
model = build_pointnet_model(num_points=N_POINTS, num_features=X_pointcloud.shape[2], num_classes=NUM_CLASSES)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005),
    metrics=['accuracy']
) 

# define callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr=1e-6,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=120,
    batch_size=64,
    callbacks=[reduce_lr, early_stop]
) 

# Save model
model.save("code/pointnet_model.h5")
