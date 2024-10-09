import open3d as o3d
import numpy as np
import os
from glob import glob
from pathlib import Path

pcd_path = '/root/dj/code/CenterPoint-KITTI/data/vod_lidar/training/velodyne'
pcd_files = glob(pcd_path + '/*.bin')


viewer = o3d.visualization.Visualizer()
viewer.create_window()
