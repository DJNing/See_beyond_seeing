# %%
# import torch
import matplotlib.pyplot as plt
import pickle 
from pathlib import Path as P
from matplotlib.patches import Rectangle as Rec
import numpy as np
from tqdm import tqdm
from vod import frame
import json
from vod.visualization.settings import label_color_palette_2d
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameLabels, FrameTransformMatrix
from vod.frame.transformations import transform_pcl
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from vod.visualization import Visualization3D
from skimage import io
from vis_tools import fov_filtering, make_vid
from glob import glob
from collections import Counter
import matplotlib.cm as cm
import os
# from vis_tools import fov_filtering
from val_path_dict import path_dict,lidar_path_dict,lidar_baseline_tags
import sys
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                # cylinder_segment = cylinder_segment.rotate(
                    # R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=True)
                cylinder_segment = cylinder_segment.rotate(
    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
    center=cylinder_segment.get_center())
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

## import from visualization_2D instead
def get_pred_dict(dt_file):
    '''
    reads results.pkl file
    returns dictionary with str(frame_id) as key, and list of strings, 
    where each string is a predicted box in kitti format.
    '''
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


def vod_to_o3d(vod_bbx,vod_calib,is_combined=False,is_label=False,draw_thick_boxes=False):
    # modality = 'radar' if is_radar else 'lidar'
    # split = 'testing' if is_test else 'training'    
    

    Classes = ['Cyclist','Pedestrian','Car','Others']



    if is_combined:
        COLOR_PALETTE = {c:(1,0,0) for c in Classes} if is_label else {c:(0,1,0) for c in Classes}

    else:
        COLOR_PALETTE = {
            'Cyclist': (1, 0.0, 0.0),
            'Pedestrian': (0.0, 1, 0.0),
            'Car': (0.0, 0.3, 1.0),
            'Others': (0.75, 0.75, 0.75)
        }



    box_list = []
    for box in vod_bbx:
        if box['label_class'] in ['Cyclist','Pedestrian','Car']:
            # Conver to lidar_frame 
            # NOTE: O3d is stupid and plots the center of the box differently,
            offset = -(box['h']/2) 
            old_xyz = np.array([[box['x'],box['y']+offset,box['z']]])
            xyz = transform_pcl(old_xyz,vod_calib.t_lidar_camera)[0,:3] #convert frame
            extent = np.array([[box['l'],box['w'],box['h']]])
            
            # ROTATION MATRIX
            rot = -(box['rotation']+ np.pi / 2) 
            angle = np.array([0, 0, rot])
            rot_matrix = R.from_euler('XYZ', angle).as_matrix()
            
            # CREATE O3D OBJECT
            obbx = o3d.geometry.OrientedBoundingBox(xyz, rot_matrix, extent.T)
            obbx.color = COLOR_PALETTE.get(box['label_class'],COLOR_PALETTE['Others']) # COLOR
            


            if draw_thick_boxes:
                points = np.asarray(obbx.get_box_points())
                lines = [
                [0, 1],[1,7],[7,2],[2,0],
                [3,6],[6,4],[4,5],[5,3],
                [0,3],[1,6],[4,7],[2,5]
            ]
                colors = [COLOR_PALETTE.get(box['label_class'],COLOR_PALETTE['Others']) for i in range(len(lines))]
                
                if box['label_class'] == 'Car':
                    r = 0.04
                else:
                    r = 0.03
                line_mesh1 = LineMesh(points, lines, colors, radius=r)
                
                box_list += [*line_mesh1.cylinder_segments]
            else:
                box_list += [obbx]
    return box_list







def get_kitti_locations(vod_data_path):
    kitti_locations = KittiLocations(root_dir=vod_data_path,
                                output_dir="output/",
                                frame_set_path="",
                                pred_dir="",
                                )
    return kitti_locations
                             
def get_visualization_data_true_frame(kitti_locations,dt_path,frame_id,is_test_set,is_combined,draw_thick_boxes=False):


    if is_test_set:
        frame_ids  = [P(f).stem for f in glob(str(dt_path)+"/*")]
        frame_data = FrameDataLoader(kitti_locations,
                                frame_id,"",dt_path)
        vod_calib = FrameTransformMatrix(frame_data)
        

    else:
        pred_dict = get_pred_dict(dt_path)
        frame_ids = list(pred_dict.keys())
        frame_data = FrameDataLoader(kitti_locations,
                                frame_id,pred_dict)
        vod_calib = FrameTransformMatrix(frame_data)

    # get pcd
    original_radar = frame_data.radar_data
    radar_points = transform_pcl(original_radar,vod_calib.t_lidar_radar)
    radar_points,flag = fov_filtering(radar_points,frame_id,is_radar=False,return_flag=True)
    lidar_points = frame_data.lidar_data 
    lidar_points = fov_filtering(lidar_points,frame_id,is_radar=False)

    
    # colors = cm.spring(original_radar[flag][:,4])[:,:3]



    # convert into o3d pointcloud object
    radar_pcd = o3d.geometry.PointCloud()
    radar_pcd.points = o3d.utility.Vector3dVector(radar_points[:,0:3])
    radar_colors = np.ones_like(radar_points[:,0:3])
    radar_pcd.colors = o3d.utility.Vector3dVector(radar_colors)
    # radar_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    lidar_pcd = o3d.geometry.PointCloud()
    lidar_pcd.points = o3d.utility.Vector3dVector(lidar_points[:,0:3])
    lidar_colors = np.ones_like(lidar_points[:,0:3])
    # lidar_colors = np.zeros_like(lidar_points[:,0:3])
    lidar_pcd.colors = o3d.utility.Vector3dVector(lidar_colors)

    
    if is_test_set:
        vod_labels = None
        o3d_labels = None 
    else:
        vod_labels = FrameLabels(frame_data.get_labels()).labels_dict
        o3d_labels = vod_to_o3d(vod_labels,vod_calib,is_combined=is_combined,is_label=True,draw_thick_boxes=draw_thick_boxes)

    vod_preds = FrameLabels(frame_data.get_predictions()).labels_dict
    o3d_predictions = vod_to_o3d(vod_preds,vod_calib,is_combined=is_combined,is_label=False,draw_thick_boxes=draw_thick_boxes)
    

    vis_dict = {
        'radar_pcd': [radar_pcd],
        'lidar_pcd': [lidar_pcd],
        'vod_predictions': vod_preds,
        'o3d_predictions': o3d_predictions,
        'vod_labels': vod_labels,
        'o3d_labels': o3d_labels,
        'frame_id': frame_id
    }
    return vis_dict

def get_visualization_data(kitti_locations,dt_path,frame_id,is_test_set,is_combined=False):


    if is_test_set:
        frame_ids  = [P(f).stem for f in glob(str(dt_path)+"/*")]
        frame_data = FrameDataLoader(kitti_locations,
                                frame_ids[frame_id],"",dt_path)
        vod_calib = FrameTransformMatrix(frame_data)
        

    else:
        pred_dict = get_pred_dict(dt_path)
        frame_ids = list(pred_dict.keys())
        frame_data = FrameDataLoader(kitti_locations,
                                frame_ids[frame_id],pred_dict)
        vod_calib = FrameTransformMatrix(frame_data)

    # print(len(frame_ids))
    



    # get pcd
    original_radar = frame_data.radar_data
    radar_points = transform_pcl(original_radar,vod_calib.t_lidar_radar)
    radar_points,flag = fov_filtering(radar_points,frame_ids[frame_id],is_radar=False,return_flag=True)
    lidar_points = frame_data.lidar_data 
    lidar_points = fov_filtering(lidar_points,frame_ids[frame_id],is_radar=True)

    
    colors = cm.spring(original_radar[flag][:,4])[:,:3]



    # convert into o3d pointcloud object
    radar_pcd = o3d.geometry.PointCloud()
    radar_pcd.points = o3d.utility.Vector3dVector(radar_points[:,0:3])
    # radar_colors = np.ones_like(radar_points[:,0:3])
    radar_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    lidar_pcd = o3d.geometry.PointCloud()
    lidar_pcd.points = o3d.utility.Vector3dVector(lidar_points[:,0:3])
    lidar_colors = np.ones_like(lidar_points[:,0:3])
    lidar_pcd.colors = o3d.utility.Vector3dVector(lidar_colors)

    
    if is_test_set:
        vod_labels = None
        o3d_labels = None 
    else:
        vod_labels = FrameLabels(frame_data.get_labels()).labels_dict
        o3d_labels = vod_to_o3d(vod_labels,vod_calib,is_combined=is_combined,is_label=True)
            

    vod_preds = FrameLabels(frame_data.get_predictions()).labels_dict
    o3d_predictions = vod_to_o3d(vod_preds,vod_calib,is_combined=is_combined,is_label=False)
    

    vis_dict = {
        'radar_pcd': [radar_pcd],
        'lidar_pcd': [lidar_pcd],
        'vod_predictions': vod_preds,
        'o3d_predictions': o3d_predictions,
        'vod_labels': vod_labels,
        'o3d_labels': o3d_labels,
        'frame_id': frame_ids[frame_id]
    }
    return vis_dict



def set_camera_position(vis_dict,output_name):


    geometries = []
    geometries += vis_dict['radar_pcd']
    geometries += vis_dict['o3d_labels']

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=656,height=403)    
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()

    o3d.io.write_pinhole_camera_parameters(f'{output_name}.json', param)
    vis.destroy_window()

def vis_one_frame(
    vis_dict,
    camera_pos_file,
    output_name,
    plot_radar_pcd=True,
    plot_lidar_pcd=False,
    plot_labels=False,
    plot_predictions=False):

    
    geometries = []
    name_str = ''

    if plot_radar_pcd:
        geometries += vis_dict['radar_pcd']
        point_size = 3
        name_str += 'Radar'
    if plot_lidar_pcd:
        geometries += vis_dict['lidar_pcd']
        point_size = 1 
        name_str += 'Lidar'
    if plot_labels:
        geometries += vis_dict['o3d_labels']
        name_str += 'GT'
    if plot_predictions:
        geometries += vis_dict['o3d_predictions']
        name_str += 'Pred'

    if name_str != '':
        output_name  = output_name / name_str
    output_name.mkdir(parents=True,exist_ok=True)


    width,height = get_resolution(camera_pos_file)
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(width=width,height=height)    
    # DRAW STUFF
    for geometry in geometries:
        viewer.add_geometry(geometry)
    
    # POINT SETTINGS
    opt = viewer.get_render_option()
    opt.point_size = point_size
    
    # BACKGROUND COLOR
    opt.background_color = np.asarray([0, 0, 0])

    # SET CAMERA POSITION
    ctr = viewer.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters(camera_pos_file)    
    ctr.convert_from_pinhole_camera_parameters(parameters)
    
    # viewer.run()
    frame_id = vis_dict['frame_id']
    viewer.capture_screen_image(f'{output_name}/{frame_id}.png',True)
    viewer.destroy_window()



def vis_all_frames(
    kitti_locations,
    dt_path,
    CAMERA_POS_PATH,
    OUTPUT_IMG_PATH,
    plot_radar_pcd,
    plot_lidar_pcd,
    plot_labels,
    plot_predictions,
    is_test_set = False,
    is_combined=False):

    
    if is_test_set:
        frame_ids  = [P(f).stem for f in glob(str(dt_path)+"/*")]

    else:
        pred_dict = get_pred_dict(dt_path)
        frame_ids = list(pred_dict.keys())

    for i in tqdm(range(len(frame_ids))):
        vis_dict = get_visualization_data(kitti_locations,dt_path,i,is_test_set,is_combined)
        vis_one_frame(
            vis_dict = vis_dict,
            camera_pos_file=CAMERA_POS_PATH,
            output_name=OUTPUT_IMG_PATH,
            plot_radar_pcd=plot_radar_pcd,
            plot_lidar_pcd=plot_lidar_pcd,
            plot_labels=plot_labels,
            plot_predictions=plot_predictions)
        # break
    

# %%

def do_vis(tag,frames,is_test_set,test_dict,vod_data_path,CAMERA_POS_PATH,bool_dict):
        abs_path = P(__file__).parent.resolve()
        base_path = abs_path.parents[1]
        # CAMERA_POS_PATH = 'widecam.json'
        output_name = tag+'_testset' if is_test_set else tag 
        OUTPUT_IMG_PATH = base_path /'output' / 'vod_vis' / 'vis_video' /   CAMERA_POS_PATH[:-5] / (output_name) 
        print(f'IMAGES WILL BE SAVED TO DIRECTORY: {OUTPUT_IMG_PATH}')

        OUTPUT_IMG_PATH.mkdir(parents=True,exist_ok=True)

        detection_result_path = base_path / path_dict[tag]

        dt_path = str(detection_result_path / 'result.pkl')    
        test_dt_path = base_path / test_dict[tag] if is_test_set else base_path / path_dict[tag]

        vis_path = test_dt_path if is_test_set else dt_path

        kitti_locations = get_kitti_locations(vod_data_path)





        vis_subset(
            kitti_locations,
            vis_path,
            CAMERA_POS_PATH,
            OUTPUT_IMG_PATH,
            frame_ids=frames,
            bool_dict = bool_dict)
        
def vis_subset(
kitti_locations,
    dt_path,
    CAMERA_POS_PATH,
    OUTPUT_IMG_PATH,
    frame_ids,
    bool_dict):


    for i in tqdm(range(len(frame_ids))):
        vis_dict = get_visualization_data_true_frame(
            kitti_locations,
            dt_path,
            frame_ids[i],
            False,
            is_combined=bool_dict['is_combined'],
            draw_thick_boxes = bool_dict['DRAW_THICK_BOXES'])


        vis_one_frame(
            vis_dict = vis_dict,
            camera_pos_file=CAMERA_POS_PATH,
            output_name=OUTPUT_IMG_PATH,
            plot_radar_pcd=bool_dict['plot_radar_pcd'],
            plot_lidar_pcd=bool_dict['plot_lidar_pcd'],
            plot_labels=bool_dict['plot_labels'],
            plot_predictions=bool_dict['plot_predictions'])



# %%
def main():
    '''
    NOTE: EVERYTHING IS PLOTTED IN THE LIDAR FRAME 
    i.e. radar,lidar,gt,pred boxes all in lidar coordinate frame 
    '''

    vod_data_path = '/mnt/12T/public/view_of_delft'

    

    # ['centerpoint_lidar','3dssd_lidar','pp_lidar','lidar_i'] 

    test_dict = {
        'CFAR_lidar_rcsv':'/root/gabriel/code/parent/CenterPoint-KITTI/output/root/gabriel/code/parent/CenterPoint-KITTI/output/IA-SSD-GAN-vod-aug-lidar/to_lidar_5_feat/IA-SSD-GAN-vod-aug-lidar/default/eval/epoch_5/val/default/final_result/data',
        'CFAR_radar':'output/root/gabriel/code/parent/CenterPoint-KITTI/output/IA-SSD-GAN-vod-aug/radar48001_512all/IA-SSD-GAN-vod-aug/default/eval/epoch_512/val/default/final_result/data',

    }
    


    
    abs_path = P(__file__).parent.resolve()
    base_path = abs_path.parents[1]
    #------------------------------------SETTINGS------------------------------------
    frame_id = 333
    resolution_dict = {
        '720': [720, 1280],
        'wide_angle2':[1080,480]
    
    }
    resolution = '480'
    is_test_set = False
    is_combined = True
    tag = 'CFAR_lidar_rcs'
    
    abs_path = P(__file__).parent.resolve()
    base_path = abs_path.parents[1]
    CAMERA_POS_PATH = 'thickcam.json'
    output_name = tag+'_testset' if is_test_set else tag 
    OUTPUT_IMG_PATH = base_path /'output' / 'vod_vis' / 'vis_video' /  resolution / (output_name)
#--------------------------------------------------------------------------------

    OUTPUT_IMG_PATH.mkdir(parents=True,exist_ok=True)

    detection_result_path = base_path / path_dict[tag]

    dt_path = str(detection_result_path / 'result.pkl')    
    test_dt_path = base_path / test_dict[tag] if is_test_set else base_path / path_dict[tag]

    vis_path = test_dt_path if is_test_set else dt_path

    kitti_locations = get_kitti_locations(vod_data_path)


    vis_all_frames(
        kitti_locations,
        vis_path,
        CAMERA_POS_PATH,
        OUTPUT_IMG_PATH,
        plot_radar_pcd=False,
        plot_lidar_pcd=True,
        plot_labels=True,
        plot_predictions=True,
        is_test_set=is_test_set,
        is_combined=is_combined)


    # # #### CREATE DETECTION DATABASE #### 
    # tag_list = ['CFAR_lidar_rcs','lidar_i']
    # dt_paths = get_paths(base_path,path_dict,tag_list)
    # compare_models(kitti_locations,dt_paths,tag_list)

    # for tag in ['centerpoint_lidar','3dssd_lidar','pp_lidar','lidar_i']:
    #     tag_list = ['CFAR_lidar_rcs',tag]
    #     dt_paths = get_paths(base_path,path_dict,tag_list)
    #     compare_models(kitti_locations,dt_paths,tag_list)
    

    # # TODO: put this into a function 
    # test_path = '/root/gabriel/code/parent/CenterPoint-KITTI/output/vod_vis/vis_video/CFAR_lidar_rcs1080_zoomedv2/LidarPred'
    # save_path = '/root/gabriel/code/parent/CenterPoint-KITTI/output/vod_vis/vis_video/CFAR_lidar_rcs1080_zoomedv2_LidarPred.mp4'
    # dt_imgs = sorted(glob(str(P(test_path)/'*.png')))
    # make_vid(dt_imgs, save_path, fps=15)





#%%


def get_filtering_data(kitti_locations,dt_path,frame_id):
    pred_dict = get_pred_dict(dt_path)
    frame_ids = list(pred_dict.keys())
    frame_data = FrameDataLoader(kitti_locations,
                            frame_ids[frame_id],pred_dict)
    vod_calib = FrameTransformMatrix(frame_data)

    # print(len(frame_ids))
 
    vod_labels = FrameLabels(frame_data.get_labels()).labels_dict
   

    vod_preds = FrameLabels(frame_data.get_predictions()).labels_dict
    

    vis_dict = {
        'vod_predictions': vod_preds, 
        'vod_labels': vod_labels,
        'frame_id': frame_ids[frame_id]
    }
    return vis_dict





def gather_frames(frame_ids,det_path1,det_path2,gt_path,output_path):

  

    for frame in frame_ids:
        path = output_path / str(frame)
        os.makedirs(path,exist_ok=True)
        img1 = det_path1 + f"/{str(frame).zfill(5)}.png"
        img2 = det_path2 + f"/{str(frame).zfill(5)}.png"
        gt = gt_path + f"/{str(frame).zfill(5)}.png"
        
        os.symlink(img1,path / P(str(P(det_path1).parents[0].stem)+".png"))
        os.symlink(img2,path / P(str(P(det_path2).parents[0].stem)+".png"))
        os.symlink(gt,path / P("gt.png"))




def analyze_models(counter_path,frame_id_path,k):

    counter = np.load(counter_path)
    frame_ids = np.load(frame_id_path)

    model_counts = counter[:,:-1,:]
    gt_count = counter[:,-1,:]

    repeated_gt = np.repeat(np.expand_dims(gt_count,axis=1),model_counts.shape[1],axis=1)
    abs_diff = np.abs(repeated_gt-model_counts)
    difference_to_gt = np.sum(abs_diff,axis=2)



    relative_diff = np.abs(difference_to_gt[:,0]-difference_to_gt[:,1])
    diff_tens = torch.from_numpy(relative_diff)
    v,i = torch.topk(diff_tens,k)
    # ind = np.argpartition(relative_diff, -k)[-k:]

    return frame_ids[i.numpy()],v

def get_paths(base_path,path_dict,tag_list):
    dt_paths = []
    for tag in tag_list:
        detection_result_path = base_path / path_dict[tag]
        dt_path = str(detection_result_path / 'result.pkl')   
        dt_paths += [dt_path]
    return dt_paths

def compare_models(
    kitti_locations,
    dt_paths,
    tag_list,
    is_test_set = False):

    
    sample_dt = dt_paths[0]
    pred_dict = get_pred_dict(sample_dt)
    frame_ids = list(pred_dict.keys())
    
    # 1296 x num_models x 3:(car,ped,cyclist) 
    counter = np.zeros((len(frame_ids),len(dt_paths)+1,3))
    name_to_int = {'Car':0,'Pedestrian':1,'Cyclist':2}
    frame_ids_np = np.array([int(f) for f in frame_ids])

    for i in tqdm(range(len(frame_ids))):
        for j,dt_path in enumerate(dt_paths):
            vis_dict = get_visualization_data(kitti_locations,dt_path,i,is_test_set)
            detected_classes = [v['label_class'] for v in vis_dict['vod_predictions']]
            class_counter = Counter(detected_classes)
            for c in class_counter:
                counter[i][j][name_to_int[c]] = class_counter[c]
                # print("")

        gt_labels= [v['label_class'] for v in vis_dict['vod_labels'] if v['label_class'] in ['Car','Pedestrian','Cyclist']]
        gt_counter = Counter(gt_labels)
        for c in gt_counter:
                counter[i][-1][name_to_int[c]] = gt_counter[c]
    
    
    with open(f'{tag_list[0]}{tag_list[1]}.npy', 'wb') as f:
        np.save(f,counter)

def get_resolution(camera_path):
    camera_params = json.load(open(camera_path))
    height = camera_params['intrinsic']['height']
    width = camera_params['intrinsic']['width']

    return width,height

def generate_comparison_frames():
    
    ###############################FRAMES####################################
    # frames = np.load('/root/gabriel/code/parent/CenterPoint-KITTI/tools/visual_utils/frames_to_generate.npy')
    # frames = ['4891','162','003','153','183','191','277','328','3593','04740','4878','05068','08433']
    frames = list(range(189,370))+list(range(4878,4987))
    frames += list(range(8420,8480))
    frames += list(range(422,495))

    frames = [str(f).zfill(5) for f in frames]
    ########################################################################

    

    #############################SETTINGS###################################
    main_tag = 'CFAR_radar'
    vod_data_path = '/mnt/12T/public/view_of_delft'
    # CAMERA_POS_PATH = 'cam_960_480_angle_2.json'
    CAMERA_POS_PATH = 'camera_960_1080.json'
    
    is_test_set = False
    draw_main_tag = False
    bool_dict = {
        'plot_radar_pcd': True,
        'plot_lidar_pcd': False,
        'plot_labels': True,
        'plot_predictions': True,
        'is_combined': True,
        'DRAW_THICK_BOXES': False
    }
    """
    NOTE: setting is_combined = True overrides the class colors.
          GT = RED
          PRED = GREEN
    """
    ########################################################################
    



    res = get_resolution(CAMERA_POS_PATH)
    all_tags = lidar_baseline_tags+[main_tag] if draw_main_tag else lidar_baseline_tags
    all_tags = ['pp_radar_rcsv']
    for tag in all_tags:
        print(f'visualizing tag: {tag} with resolution {res[0]}x{res[1]}')        
        do_vis(
            tag,
            frames,
            is_test_set,
            None,
            vod_data_path,
            CAMERA_POS_PATH,
            bool_dict)
    


def concat_two_image(img1,img2,out_name):

    
    images = [Image.open(x) for x in [img1,img2]]

    widths, heights = zip(*(i.size for i in images))
    
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(f'{out_name}.png')   



def concat_outputs(path1,path2):

    frame_ids = sorted(glob(path1+"/*"))
    frame_id2 = sorted(glob(path2+"/*"))


    assert len(frame_ids)==len(frame_id2)
    
    for i in tqdm(range(len(frame_ids))):
        parent_path = P(frame_ids[i]).parents[2]
        frame_id = P(frame_ids[i]).stem
        out_name = f'{str(parent_path)}/' + str(P(frame_ids[i]).parents[1].stem) + str(P(frame_id2[i]).parents[1].stem) + f'/{str(frame_id)}'
        # print(parent_path,frame_id)
        # print(out_name)
        # print(frame_ids[i],frame_id2[i])
        os.makedirs(f'{str(parent_path)}/' + str(P(frame_ids[i]).parents[1].stem) + str(P(frame_id2[i]).parents[1].stem),exist_ok=True)
        concat_two_image(frame_ids[i],frame_id2[i],out_name)



def create_video(image_path,save_path):



    # test_path = '/root/gabriel/code/parent/CenterPoint-KITTI/output/vod_vis/vis_video/CFAR_lidar_rcs1080_zoomedv2/LidarPred'
    # save_path = '/root/gabriel/code/parent/CenterPoint-KITTI/output/vod_vis/vis_video/CFAR_lidar_rcs1080_zoomedv2_LidarPred.mp4'
    dt_imgs = sorted(glob(str(P(image_path)/'*.png')))
    make_vid(dt_imgs, save_path, fps=10)


def add_captions(image_path,save_path,caption1,caption2):
    W, H = 1920,1080
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("cmunss.ttf", 64)
    _, _, w, h = draw.textbbox((0, 0), caption1, font=font)
    draw.text((((W-w)/4), ((H-h)*0.95)), caption1, font=font, fill=(255,255,255))
    _, _, w2, h2 = draw.textbbox((0, 0), caption2, font=font)
    draw.text((((W-w2)*0.75), ((H-h2)*0.95)), caption2, font=font, fill=(255,255,255))

    img.save(save_path)

    



#%%
if __name__ == "__main__":
    '''
    WORKFLOW: 
    1. This file ONLY works when cd into tools/eval_utils (because of o3d heading rendering)
    2a. Run: `source py3env/bin/activate`
    2b. Run: `export PYTHONPATH="${PYTHONPATH}:<REPO ROOT DIRECTORY>"`
        for me,  <REPO ROOT DIRECTORY> = /root/gabriel/code/parent/CenterPoint-KITTI
    3. Choose the correct settings (lines 749-770)
    4. Finally Run `python draw_3d.py`
    
    
    Note:
    main() is depreciated for now
    '''
    

    # RUN THESE COMMANDS FIRST
    # source py3env/bin/activate
    # export PYTHONPATH="${PYTHONPATH}:/root/gabriel/code/parent/CenterPoint-KITTI"
    
    

    # generate_comparison_frames()

    add_captions(
        image_path='/root/gabriel/code/parent/CenterPoint-KITTI/output/vod_vis/vis_video/camera_960_1080/CFAR_radarpp_radar_rcsv/00189.png',
        save_path='/root/gabriel/code/parent/CenterPoint-KITTI/output/vod_vis/vis_video/camera_960_1080/CFAR_radarpp_radar_rcsv_captioned/00189.png',
        caption1='Ours',
        caption2='Pointpillar'
    )   

    # concat_outputs(
    #     path1 = '/root/gabriel/code/parent/CenterPoint-KITTI/output/vod_vis/vis_video/camera_960_1080/CFAR_radar/RadarGTPred',
    #     path2 = '/root/gabriel/code/parent/CenterPoint-KITTI/output/vod_vis/vis_video/camera_960_1080/pp_radar_rcsv/RadarGTPred'
    # )

    # create_video(
    #     image_path='/root/gabriel/code/parent/CenterPoint-KITTI/output/vod_vis/vis_video/camera_960_1080/CFAR_lidar_rcspp_lidar',
    #     save_path='/root/gabriel/code/parent/CenterPoint-KITTI/output/vod_vis/vis_video/camera_960_1080/CFAR_lidar_rcspp_lidar.mp4'
    # )




# %%
