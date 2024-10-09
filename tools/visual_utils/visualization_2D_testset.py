# %%
# from vod import frame
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameLabels, FrameTransformMatrix
# from vod.frame.transformations import homogeneous_coordinates, homogeneous_transformation
# from vod.visualization.helpers import get_transformed_3d_label_corners,get_3d_label_corners
from vod.visualization import Visualization2D
# from mpl_toolkits import mplot3d
import os 
import numpy as np
# import matplotlib.pyplot as plt
import pickle
from pathlib import Path as P


def get_pred_dict(dt_file):
    '''
    reads results.pkl file

    returns dictionary with str(frame_id) as key, and list of strings, 
    where each string is a predicted box in kitti format.
    '''
    dt_file = dt_file
    dt_annos = []

    # load detection dict
    with open(dt_file, 'rb') as f:
        infos = pickle.load(f)
        dt_annos.extend(infos)      

    labels_dict = {}
    for j in range(len(dt_annos)):
        labels = []
        curr = dt_annos[j]
        frame_id = curr['frame_id']
        
        # no predicted 
        if len(dt_annos[j]['name']) == 0: 
            labels += []
        
        else:
            for i in range(len(dt_annos[j]['name'])):       
                # extract the relevant info and format it 
                line = [str(curr[x][i]) if not isinstance(curr[x][i],np.ndarray) else [y for y in curr[x][i]]  for x in list(curr.keys())[:-2]]
                flat = [str(num) for item in line for num in (item if isinstance(item, list) else (item,))]
                
                # L,H,W -> H,W,L 
                flat[9],flat[10] = flat[10],flat[9]
                flat[8],flat[10] = flat[10],flat[8]
                
                labels += [" ".join(flat)]

        labels_dict[frame_id] = labels
    
    return labels_dict






# %%
from pathlib import Path


path_dict = {
    'pp_radar': 'output/root/gabriel/detections/pp_radar',
    'iassd_radar': 'output/root/gabriel/detections/iassd_radar',
    'CFAR_radar': 'output/root/gabriel/detections/cfar_radar',
}
abs_path = P(__file__).parent.resolve()
base_path = abs_path.parents[1]


#### CHOOSE HERE ####
key = 'pp_radar'
output_name = 'vod_gt'

op_dir = os.path.join(base_path,'output/vod_vis',output_name+"/")
op_vid = os.path.join(base_path,'output/vod_vis',output_name+".mp4")
detection_path = os.path.join(base_path,path_dict[key])
##################



# folder containing vod_lidar and vod_radar 
root_dir = '/mnt/12T/public/view_of_delft' 
# TODO: just read the file names
test_result_path = '/root/gabriel/code/parent/CenterPoint-KITTI/output/root/gabriel/code/parent/CenterPoint-KITTI/output/IA-SSD-GAN-vod-aug/radar48001_512all/IA-SSD-GAN-vod-aug/default/eval/epoch_512/val/default/result.pkl'
pred_dict = get_pred_dict(test_result_path)
frame_ids = list(pred_dict.keys())




temp = Path(op_dir).mkdir(exist_ok=True)
kitti_locations = KittiLocations(root_dir=root_dir,
                                output_dir=op_dir,
                                frame_set_path="",
                                pred_dir="",
                                )
#%%
from tqdm import tqdm
# type frame u want to see here instead of "00334", or index into frame_ids[0...1295]
for id in tqdm(frame_ids):
    frame_data = FrameDataLoader(kitti_locations,
                                id,
                                test_set_path=detection_path)

    vis2d = Visualization2D(frame_data)
    vis2d.draw_plot(
        show_radar=True,
        show_lidar=False,
        show_gt=True,
        show_pred=False,
        plot_figure=False)



#%% 
from vis_tools import make_vid
from glob import glob

imgs = sorted(glob(op_dir+'*.png'))

make_vid(imgs, op_vid, fps=10)








# %%
