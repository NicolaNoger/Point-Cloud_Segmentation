import os
import numpy as np
import tensorflow as tf
from pointnet2.models import pointnet2_sem_seg
from pointnet2.utils import provider


# === KONFIGURATION ===
NUM_CLASSES = 9  # z. B. Wasser, Boden, Vegetation, Gebäude, etc.
NUM_POINT = 4096
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

TRAIN_DIR = '/path/to/npy/train'
VAL_DIR = '/path/to/npy/val'


# === DATENLADEN ===
def load_npy_dataset(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".npy")]
    def generator():
        for file in files:
            data = np.load(file, allow_pickle=True).item()
            yield data['points'], data['labels']

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(NUM_POINT, None), dtype=tf.float32),
            tf.TensorSpec(shape=(NUM_POINT,), dtype=tf.int32)
        )
    )


def prepare_dataset(ds):
    return (ds
        .shuffle(100)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE))

train_ds = prepare_dataset(load_npy_dataset(TRAIN_DIR))
val_ds = prepare_dataset(load_npy_dataset(VAL_DIR))


# === MODELL ===
model = pointnet2_sem_seg.get_model(NUM_CLASSES)
loss_fn = pointnet2_sem_seg.get_loss()

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)


# === TRAININGSLOOP ===
@tf.function
def train_step(points, labels):
    with tf.GradientTape() as tape:
        logits = model(points, training=True)
        loss = loss_fn(logits, labels)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def val_step(points, labels):
    logits = model(points, training=False)
    loss = loss_fn(logits, labels)
    return loss

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_loss = []
    for points, labels in train_ds:
        loss = train_step(points, labels)
        train_loss.append(loss.numpy())

    val_loss = []
    for points, labels in val_ds:
        loss = val_step(points, labels)
        val_loss.append(loss.numpy())

    print(f"Train loss: {np.mean(train_loss):.4f} | Val loss: {np.mean(val_loss):.4f}")