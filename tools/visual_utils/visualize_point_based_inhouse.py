# %%
import matplotlib.pyplot as plt
import pickle 
from pathlib import Path as P
from matplotlib.patches import Rectangle as Rec
import numpy as np
from pcdet.utils import calibration_kitti
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tqdm import tqdm
from vod.visualization.settings import label_color_palette_2d
from scipy.spatial.transform import Rotation as R


def transform_anno(loc, frame_id, is_radar=True, is_test=False):
    x, y, z = loc[0], loc[1], loc[2]
    modality = 'radar' if is_radar else 'lidar'
    split = 'testing' if is_test else 'training'

    calib_path = "../../data/vod_%s/%s/calib/%s.txt"%(modality, split, frame_id)

    # if is_radar:
    #     calib_path = "/root/dj/code/CenterPoint-KITTI/data/vod_radar/training/calib/{0}.txt".format(frame_id)
    # else:
    #     calib_path = "/root/dj/code/CenterPoint-KITTI/data/vod_lidar/training/calib/{0}.txt".format(frame_id)
    
    
    # COMMENTING FOR INHOUSE
    
    # calib = calibration_kitti.Calibration(calib_path)
    # loc = np.array([[x,y,z]])
    # loc_lidar = calib.rect_to_lidar(loc)
    # x,y,z = loc_lidar[0]
    return x,y,z


def get_rot_corner(x,y,l,w,a):

    s,c = np.sin(a),np.cos(a)

    corner_x = x - l/2
    corner_y = y - w/2

    corner_x -= x
    corner_y -= y

    new_corner_x = corner_x*c - corner_y*s 
    new_corner_y = corner_x*s + corner_y*c

    new_corner_x += x
    new_corner_y += y

    return new_corner_x,new_corner_y




def anno2plt(anno, color_dict, lw, frame_id, xz=False, is_radar=True, is_test=False):
    dim = anno['dimensions']
    loc = anno['location']

    # angle = anno['rotation_y'] * 180 / 3.14
    angle = -(anno['rotation_y']+ np.pi / 2) 
    angle = anno['rotation_y']
    rec_list = []
    cls = anno['name']
    for idx in range(dim.shape[0]):
        name = cls[idx]
        # print(name)
        if name not in color_dict:
            color = 'gray'
        else:
            color = color_dict[name]
            # print(color)
    
        if xz:

            x, _, y = transform_anno(loc[idx], frame_id, is_radar=is_radar, is_test=is_test)
            # w, _, l = dim[idx]
            l, w, _ = dim[idx]  # 
            ang = -angle[idx]* 0
        else:
            # print(loc[idx])
            x, y, z = transform_anno(loc[idx], frame_id, is_radar=is_radar, is_test=is_test)
            # print(x,y)
            
            ### X -> LENGTH
            ### Y -> WIDTH 
            ### Z -> HEIGHT, not used. 
            # x,y,z = loc[idx]
            # w, l, _ = dim[idx]
            # print(dim[idx]) 
            l, h, w  = dim[idx] # <-- SHOULD BE CORRECT? 
            ang = angle[idx]
            # ang = 0
            # print(l,w,ang)
            # print("="*40)

            ax,ay = get_rot_corner(x,y,l,w,ang)
            ang = ang * 180 / 3.14
            # ax = x - (l/4)
            # ay = y - (w/4)

        rec_list += [Rec((ax, ay), l, w, ang, fill=False, color=color,lw=lw)]
    return rec_list



def drawBEV(ax, pts, centers, annos, color_dict, frame_id, ax_title, ext_legends=[], is_radar=True, is_test=False):


    # 3. draw bbx
    try:
        rec_list = anno2plt(annos, color_dict, 2, frame_id=frame_id, xz=False, is_radar=is_radar, is_test=is_test)
    except:
        rec_list = anno2plt(annos[0], color_dict, 2, frame_id=frame_id, xz=False, is_radar=is_radar, is_test=is_test)
    
    # 1. draw original points if exist
    if pts is not None:
        x = pts[:, 1]
        y = pts[:, 2]
        ax.scatter(x, y, c='black', s=0.1)
    # 2. overlay centers
    if centers is not None:
        cx = centers[:, 1]
        cy = centers[:, 2]
        ax.scatter(cx, cy, c='red', s=0.1)
    
    for rec in rec_list:
        ax.add_patch(rec)

    # legend_elements = [Patch(facecolor='white', edgecolor=v, label=k) for i, (k, v) in enumerate(color_dict.items())]
    legend_elements = [Patch(facecolor='white', edgecolor='blue', label='Pred'), Patch(facecolor='white', edgecolor='red', label='GT')]
    if centers is not None:
        legend_elements += [Line2D([0], [0], marker='o', color='w', label='FG points',
                          markerfacecolor='r', markersize=10)]
    legend_elements += ext_legends
    ax.legend(handles=legend_elements, loc=1)
    ax.set_title(ax_title)


def draw_rectangle(ax, anno, color_dict, xz=False):
    recs = anno2plt(anno, color_dict, lw=2, xz=xz)
    for rec in recs:
        ax.add_patch(rec)

def draw_point_cloud(ax, pc, c='black',s=0.1):
    ax.scatter(pc[:, 0], pc[:, 1], c=c, s=s)

def get_matrix_from_ext(ext):
    rot = R.from_euler('ZYX', ext[3:], degrees=True)
    rot_m = rot.as_matrix()
    x, y, z = ext[:3]
    tr = np.eye(4)
    tr[:3,:3] = rot_m
    tr[:3, 3] = np.array([x, y, z]).T
    return tr

if __name__ == '__main__':
    
    # cls_name = ['Car','Pedestrian', 'Cyclist', 'Others']
    cls_name = ['Car','Truck']
    
    # lidar pointcloud path
    lidar_pc_path = P('/root/hantao/CenterPoint-KITTI/data/shangqi_new_lidar/training/velodyne')
    
    # radar pointcloud path
    radar_pc_path = P('/root/hantao/CenterPoint-KITTI/data/shangqi_new_radar/training/velodyne')

    # CHANGE PATH 
    # root_path = P('/root/hantao/CenterPoint-KITTI/output/pointpillar_inhouse_new_lidar/initial_pct_0401/eval/checkpoint_epoch_80')
    # root_path = P('/root/hantao/CenterPoint-KITTI/output/CFAR-lidar-new-inhouse/initial_pct_0401/eval/checkpoint_epoch_20')

    # root_path = P('/root/hantao/CenterPoint-KITTI/output/pointpillar_inhouse_new_radar/initial_pct_0401/eval/checkpoint_epoch_80')
    # root_path = P('/root/hantao/CenterPoint-KITTI/output/IA-SSD-new-inhouse-radar/initial_pct_0401/eval/checkpoint_epoch_69')
    # root_path = P('/root/hantao/CenterPoint-KITTI/output/CFAR-radar-new-inhouse/initial_pct_0401/eval/checkpoint_epoch_4')

    # root_path = P('/root/hantao/CenterPoint-KITTI/output/pointpillar_inhouse_new_lidar/initial_pct_0401/eval/checkpoint_epoch_74')
    # root_path = P('/root/hantao/CenterPoint-KITTI/output/IA-SSD-new-inhouse-lidar/initial_pct_0401/eval/checkpoint_epoch_75')
    root_path = P('/root/hantao/CenterPoint-KITTI/output/CFAR-lidar-new-inhouse/initial_pct_0401/eval/checkpoint_epoch_17')
    color_dict = {}

    gt_save_dir = root_path / 'GT_all_bev'
    pred_save_dir = root_path / 'pred_bev'
    gt_save_dir.mkdir(exist_ok=True)
    pred_save_dir.mkdir(exist_ok=True)

    GT_color = {}

    for i, v in enumerate(cls_name):
        color_dict[v] = label_color_palette_2d[v]
        # GT COLOR 
        GT_color[v] = 'red'

    with open(str(root_path / 'gt.pkl'), 'rb') as f:
        gt = pickle.load(f)

    # load det
    with open(str(root_path / 'dt.pkl'), 'rb') as f:
        dt = pickle.load(f)


    data_dict = {}

    plt.rcParams['figure.dpi'] = 150
    # fig, ax = plt.subplots()
    plt.xlim(-0,75)
    plt.ylim(-30,30)
    ax = plt.gca()

    # 1. image with center origin and gt annos
    ids = list(gt.keys())
    print('=====> drawing detection BEVs')
    for id in tqdm(ids):
        img_fname = str(id) + '.png'
        
        # print(id,points[id])
        # draw gt
        drawBEV(ax, None, None, gt[id], GT_color, id, 'GT')
        # gt_img_full_fname = str(gt_save_dir / img_fname)
        # plt.xlim(-0,75)
        # plt.ylim(-30,30)
        # plt.savefig(gt_img_full_fname)
        # ax.clear()
        
        # draw pred
        title = str(root_path).split('/')[-4]
        drawBEV(ax, None,None, dt[id], color_dict, id, title)
        # drawBEV(ax, points[id], centers_origin[id], dt[id], color_dict, id, 'pred')

        pc_bin = str(lidar_pc_path/(id + '.bin'))
        pc = np.fromfile(str(pc_bin), dtype=np.float32).reshape(-1, 4)
        # pc = pc[:, :3]
        # lidar_ext = [-2.5, 0, 2.03, 4.9, -1.5, 0]
        # lidar_tr = get_matrix_from_ext(lidar_ext)
        # pc = np.hstack((pc, np.ones((pc.shape[0], 1),dtype=np.float32)))
        # pc = np.dot(pc, lidar_tr.T)
        # pc = pc[:, :3]
        draw_point_cloud(ax, pc)
        
        pc_bin = str(radar_pc_path/(id + '.bin'))
        pc = np.fromfile(str(pc_bin), dtype=np.float32).reshape(-1, 6)
        draw_point_cloud(ax, pc, c='green',s=2.0)

        pred_img_full_fname = str(pred_save_dir / img_fname)
        plt.xlim(-0,75)
        plt.ylim(-30,30)
        plt.savefig(pred_img_full_fname)
        ax.clear()
        
    
    # plt.show()


# %%
