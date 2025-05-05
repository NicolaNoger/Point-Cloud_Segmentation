import os
import numpy as np
from tqdm import tqdm
import laspy
import pandas as pd
import open3d as o3d
import numpy as np



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
        raise ValueError("Keine Punktdaten gefunden, prüfe deine .las/.laz-Dateien.")

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


def model_3d(df, point_size=1, dtype="laz"):
    """
    Creates a 3D point cloud visualization.

    Parameters:
        df: pandas.DataFrame: DataFrame containing point cloud data.
            Must contain columns for x, y, z coordinates and classification labels.
        point_size (float, optional): Size of the points in the 3D visualization.
        dtype (str, optional): Type of data. Can be "las" or "laz". (defines the color scheme)
            Default is "laz".
    """

    if dtype == "las":
        color_map = {1: [0.5, 0.5, 0.5],   
                        2: [0.33, 0.42, 0.18], 
                        3: [0.0, 0.39, 0.0],   
                        6: [0.6, 0.3, 0.0],    
                        9: [0.0, 0.0, 1.0],   
                        17: [0.4, 0.26, 0.13] 
                        }
    elif dtype == "laz": 
        color_map = {
    1: [0.6, 0.6, 0.6],  # Übriges: Grau
    2: [0.2, 0.6, 0.2],  # Bodenpunkte / DOM + DTM: Dunkelgrün
    3: [0.0, 0.8, 0.0],  # Vegetation tief (< .5m) / DOM: Helles Grün
    4: [0.0, 0.6, 0.0],  # Vegetation mittel (< 3m) / DOM: Mittelgrün
    5: [0.0, 0.4, 0.0],  # Vegetation hoch (> 3m) / DOM: Dunkelgrün
    6: [0.6, 0.3, 0.0],  # Gebäude / DOM: Braun
    7: [0.8, 0.0, 0.0],  # Ausreißer und Fehlmessungen: Rot
    10: [0.0, 0.0, 0.6], # Brücken und Stege / DOM: Blau
    12: [0.6, 0.6, 0.0], # Streifenrandpunkte: Gelb
    15: [0.9, 0.6, 0.0], # Freileitungen, Masten und Antennen: Orange
}



    # Spaltennamen in Kleinbuchstaben mappen
    cols = {col.lower(): col for col in df.columns}

    required = ["x", "y", "z", "classification"]
    if not all(k in cols for k in required):
        raise ValueError(f"DataFrame muss die Spalten {required} enthalten (Groß-/Kleinschreibung egal).")

    xyz = df[[cols["x"], cols["y"], cols["z"]]].to_numpy()
    labels = df[cols["classification"]].astype(int).to_numpy()
    colors = np.array([color_map.get(lbl, [1, 1, 1]) for lbl in labels])  # Fallback: weiß

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = point_size

    vis.run()
    vis.destroy_window()

