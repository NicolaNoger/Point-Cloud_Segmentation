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


def concat_pointcloud(directory, chunk_size=100000, save_npy=False, out_dir="processed", tile_name="all_tiles"):
    """
        Loads multiple .las files, processes them in manageable chunks, and aggregates
        the results into a single NumPy array and optionally saves it.

        Arguments: 
            directory:      Path to the data
            chunk_size:     Number of points processed at a time
            save_npy:       If True, saves .npy files for points and labels
            out_dir:        Output directory for .npy files
            tile_name:      Basename for saved files

        Returns:
            df:             DataFrame with all points
            points_array:   NumPy array with all points
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if f.endswith(".las") or f.endswith(".laz")]
    all_points = []
    header = []

    for file in tqdm(files, desc="Lade .las-Dateien"):
        las = laspy.read(file)
        dims = list(las.points.point_format.dimension_names)
        scale = las.header.scale
        offset = las.header.offset
        if not header:
            header = dims
        total_points = len(las.points)

        start = 0
        while start < total_points:
            end = min(start + chunk_size, total_points)
            chunk_data = []
            for d in dims:
                data_array = np.array(las.points[d][start:end], dtype=float)
                if d in ["X", "Y", "Z"]:
                    idx = ["X", "Y", "Z"].index(d)
                    data_array = data_array * scale[idx] + offset[idx]
                if data_array.size == 0:
                    break
                chunk_data.append(data_array)
            if len(chunk_data) != len(dims):
                start = end
                continue
            chunk = np.vstack(chunk_data).T
            all_points.append(chunk)
            start = end

    if not all_points:
        raise ValueError("Keine Punktdaten gefunden prüfe deine .las/.laz-Dateien.")

    points_array = np.vstack(all_points)
    df = pd.DataFrame(points_array, columns=header)

    # Optional: .npy speichern
    if save_npy:
        os.makedirs(out_dir, exist_ok=True)
        
        # Feature-Columns, z. B. ohne gps_time oder user_data (falls nicht gebraucht)
        try:
            feature_cols = ["X", "Y", "Z", "intensity", "return_number", "number_of_returns"]
            label_col = "classification"
            features = df[feature_cols].to_numpy()
            labels = df[label_col].to_numpy().astype(np.int32)

            np.save(os.path.join(out_dir, f"{tile_name}_points.npy"), features)
            np.save(os.path.join(out_dir, f"{tile_name}_labels.npy"), labels)
            print(f"Gespeichert: {tile_name}_points.npy und {tile_name}_labels.npy")
        except KeyError as e:
            print(f"Fehlende Spalte zum Speichern: {e}")

    return df, points_array




