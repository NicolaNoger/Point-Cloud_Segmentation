import os
import numpy as np
from tqdm import tqdm
import laspy

def las_concat(directory, chunk_size=100000):
    """
    Loads multiple .las files, processes them in manageable chunks, and aggregates
    the results into a single NumPy array for further analysis.

    Arguments: 
        directory:      Path to the data
        chunk_size:     How many Points are processed at a time, default is 100000
    """

    las_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".las")]
    all_points = []

    for file in tqdm(las_files, desc="Loading .las files"):
        las = laspy.read(file)
        dims = las.points.point_format.dimension_names
        total_points = len(las.x)
        for start in range(0, total_points, chunk_size):
            end = min(start + chunk_size, total_points)
            chunk_data = [las[d][start:end] for d in dims]
            chunk = np.vstack(chunk_data).T
            all_points.append(chunk)

    return np.vstack(all_points)

def laz_concat(directory, chunk_size=100000):


    las_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".laz")]
    all_points = []
    priority_dims = ["x", "y", "z", "classification"]

    for file in tqdm(las_files, desc="Loading .las files"):
        las = laspy.read(file, laz_backend="pyarrow")
        dims = las.points.point_format.dimension_names

        # Sortiere Dimensionen: erst Prioritäten, dann Rest
        sorted_dims = [d for d in priority_dims if d in dims] + [d for d in dims if d not in priority_dims]

        total_points = len(las.points)
        for start in range(0, total_points, chunk_size):
            end = min(start + chunk_size, total_points)
            chunk_data = [las[d][start:end] for d in sorted_dims]
            chunk = np.vstack(chunk_data).T
            all_points.append(chunk)

    return np.vstack(all_points)



