import open3d as o3d
import numpy as np



def model_3d(points, point_size=1, color_map= {1: [0.5, 0.5, 0.5],   
                                               2: [0.33, 0.42, 0.18], 
                                               3: [0.0, 0.39, 0.0],   
                                               6: [0.6, 0.3, 0.0],    
                                               9: [0.0, 0.0, 1.0],   
                                               17: [0.4, 0.26, 0.13] 
                                               }):
    """
    Creates a 3D point cloud visualization.

    Parameters:
        points (np.ndarray): Nx4 array where columns 0-2 represent x, y, z coordinates 
                             and column 3 is the classification label.
        point_size (float, optional): Size of the points in the 3D visualization.
        color_map (dict, optional): Maps classification labels (int) to [R, G, B] values.
    """

    xyz = points[:, :3]
    labels = points[:, 3].astype(int)  # Klassifizierung nutzen

    colors = np.array([color_map.get(lbl, [1, 1, 1]) for lbl in labels])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualizer mit kleineren Punkten
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    
    # Punktgröße anpassen
    opt = vis.get_render_option()
    opt.point_size = point_size # Kleinere Punkte für feinere Details
    
    vis.run()
    vis.destroy_window()
