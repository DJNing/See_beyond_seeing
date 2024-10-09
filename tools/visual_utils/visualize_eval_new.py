import os
from turtle import color
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path as P
import pickle
from matplotlib.patches import Rectangle as Rec
import cv2


from pcdet.utils import calibration_kitti

root_dir = P('/root/dj/code/CenterPoint-KITTI/output/IA-SSD-GAN-vod/domain_cross_init_car/eval/eval_with_train/epoch_80/val')
frame_ids = root_dir / 'frame_ids.txt'
frame_ids = np.loadtxt(frame_ids, delimiter=',', dtype=str)
dt_annos = []
gt_annos = []

dt_file = root_dir / 'result.pkl'
gt_file = root_dir / 'gt.pkl'

with open(dt_file, 'rb') as f:
    infos = pickle.load(f)
    dt_annos.extend(infos)

with open(gt_file, 'rb') as f:
    infos = pickle.load(f)
    gt_annos.extend(infos)

radar_pcd_path = P('/root/dj/code/CenterPoint-KITTI/data/vod_radar/training/velodyne')

def transform_anno(loc, frame_id):
    x,y, z = loc[0], loc[1], loc[2]
    calib_path = "/root/dj/code/CenterPoint-KITTI/data/vod_radar/training/calib/{0}.txt".format(frame_id)
    calib = calibration_kitti.Calibration(calib_path)
    loc = np.array([[x,y,z]])
    loc_lidar = calib.rect_to_lidar(loc)
    x,y,z = loc_lidar[0]
    return x,y,z

def process_angle(r):

    r = -r
    if r > np.pi:
        r = r - np.pi * 2
    elif r < -np.pi:
        r = r + np.pi * 2

    return r

def getpcd(fname):
    return np.fromfile(str(fname), dtype=np.float32).reshape(-1, 7)
def anno2plt(anno, color, lw, frame_id, xz=False):
    dim = anno['dimensions']
    loc = anno['location']
    angle = anno['rotation_y'] * 180 / 3.14
    rec_list = []
    for idx in range(dim.shape[0]):
        if xz:
            x, _, y = transform_anno(loc[idx], frame_id)
            w, _, l = dim[idx]
            ang = -angle[idx]* 0
        else:
            x, y, _ = transform_anno(loc[idx], frame_id)
            w, l, _ = dim[idx]
            ang = -angle[idx]

        rec_list += [Rec((x, y), w, l, ang, fill=False, color=color,lw=lw)]
    return rec_list

def bin2pts(fname, xz=True):
    pts = getpcd(fname)
    if xz:
        pts_xy = pts[:,[0,2]]
        return pts_xy
    else:
        pts_xy = pts[:, :2]
        return pts_xy

def drawBEV(pcd_fname, dt_anno, gt_anno, save_name, cur_id, xz=False):
    pts_2d = bin2pts(pcd_fname, xz=True)
    dt_rec_list = anno2plt(dt_anno, 'red', frame_id=cur_id,lw=3, xz=xz)
    gt_rec_list = anno2plt(gt_anno, 'green', frame_id=cur_id, lw=2, xz=xz)
    x = pts_2d[:,0]
    y = pts_2d[:,1]
    plt.scatter(x, y, c='black', s=0.1)
    plt.xlim(-0,75)
    plt.ylim(-30,30)
    for dt_rec in dt_rec_list:
        plt.gca().add_patch(dt_rec)

    for gt_rec in gt_rec_list:
        plt.gca().add_patch(gt_rec)
    plt.savefig(save_name)


save_dir = root_dir/'bev_img_filtered2'
save_dir.mkdir(exist_ok=True)


plt.figure()
from tqdm import tqdm
for idx in tqdm(range(len(frame_ids))):
    cur_id = frame_ids[idx]
    cur_bin = str(radar_pcd_path/(cur_id + '.bin'))
    try:
        dt_anno = dt_annos[idx]
        gt_anno = gt_annos[idx]
    except:
        continue
    save_name = str(save_dir/(cur_id + '.png'))
    drawBEV(cur_bin, dt_anno=dt_anno, gt_anno=gt_anno, save_name=save_name, xz=True, cur_id=cur_id)
    plt.clf()

