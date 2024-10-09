from posixpath import abspath
import numpy as np
from pathlib import Path as P
import pickle
# from visualize_utils import make_vid
import cv2
from vod.visualization.settings import label_color_palette_2d
from matplotlib.lines import Line2D
import os 
from visualize_point_based import drawBEV
import matplotlib.pyplot as plt
from tqdm import tqdm

from glob import glob
import argparse

def saveODImgs(frame_ids, anno, data_path, img_path, color_dict, is_radar=True, title='pred', limit_range=None, is_test=False):
    print('=================== drawing images ===================')
    plt.rcParams['figure.dpi'] = 150
    for fid in tqdm(frame_ids):
        pcd_fname = data_path / (fid + '.bin')
        vis_pcd = get_radar(pcd_fname) if is_radar else get_lidar(pcd_fname, limit_range=limit_range)
        vis_pcd = pcd_formating(vis_pcd)
        ax = plt.gca()
        drawBEV(ax, vis_pcd, None, anno[fid], color_dict, fid, title, is_radar=is_radar, is_test=is_test)
        plt.xlim(-0,75)
        plt.ylim(-30,30)
        img_fname = img_path / (fid + '.png')
        plt.savefig(str(img_fname))
        plt.cla()

def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
            & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4]) \
            & (points[:, 2] >= limit_range[2]) & (points[:, 2] <= limit_range[5])
    return mask

def get_radar(fname):
    assert fname.exists()
    radar_point_cloud = np.fromfile(str(fname), dtype=np.float32).reshape(-1, 7)
    return radar_point_cloud

def get_lidar(fname, limit_range):
    assert fname.exists()
    lidar_point_cloud = np.fromfile(str(fname), dtype=np.float32).reshape(-1, 4)
    if limit_range is not None:
        mask = mask_points_by_range(lidar_point_cloud, limit_range)
        return lidar_point_cloud[mask]
    else:
        return lidar_point_cloud

def pcd_formating(pcd):
    num_pts = pcd.shape[0]
    zeros_pad = np.zeros([num_pts, 1])
    final_pcd = np.concatenate((zeros_pad, pcd), axis=1)
    return final_pcd

def make_vid(imgs, vid_fname, fps=15):
    print('=================== making videos ===================')
    out = None
    for fname in tqdm(imgs):
        i = cv2.imread(fname)
        if out is None:
            h, w, _ = i.shape
            size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(vid_fname), fourcc, fps, size)
            
        out.write(i)
    out.release()
    

if __name__ == '__main__':

    abs_path = P(__file__).parent.resolve()
    base_path = abs_path.parents[1]
    path_dict = {
        'CFAR_radar':'output/IA-SSD-GAN-vod-aug/radar48001_512all/eval/best_epoch_checkpoint',
        'radar_rcsv':'output/IA-SSD-vod-radar/iassd_best_aug_new/eval/best_epoch_checkpoint',
        'radar_rcs':'output/IA-SSD-vod-radar/iassd_rcs/eval/best_epoch_checkpoint',
        'radar_v':'output/IA-SSD-vod-radar/iassd_vcomp_only/eval/best_epoch_checkpoint',
        'radar':'output/IA-SSD-vod-radar-block-feature/only_xyz/eval/best_epoch_checkpoint',
        'lidar_i':'output/IA-SSD-vod-lidar/all_cls/eval/checkpoint_epoch_80',
        'lidar':'output/IA-SSD-vod-lidar-block-feature/only_xyz/eval/best_epoch_checkpoint',
        'CFAR_lidar_rcsv':'output/IA-SSD-GAN-vod-aug-lidar/to_lidar_5_feat/eval/best_epoch_checkpoint',
        'CFAR_lidar_rcs':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_rcs_only/eval/best_epoch_checkpoint',
        'CFAR_lidar_v':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_vcomp_only/eval/best_epoch_checkpoint',
        'CFAR_lidar':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_xyz_only/eval/best_epoch_checkpoint',
        ## other baselines
        'pp_radar_rcs' : 'output/pointpillar_vod_radar/debug_new/eval/checkpoint_epoch_80',
        'pp_radar_rcsv' : 'output/pointpillar_vod_radar/vrcomp/eval/best_epoch_checkpoint', 
        '3dssd_radar_rcs': 'output/3DSSD_vod_radar/rcs/eval/best_epoch_checkpoint',
        '3dssd_radar_rcsv': 'output/3DSSD_vod_radar/vcomp/eval/best_epoch_checkpoint',
        'centerpoint_radar_rcs': 'output/centerpoint_vod_radar/rcs/eval/best_epoch_checkpoint',
        'centerpoint_radar_rcsv': 'output/centerpoint_vod_radar/rcsv/eval/best_epoch_checkpoint',
        'second_radar_rcs': 'output/second_vod_radar/radar_second_with_aug/eval/checkpoint_epoch_80'.
        'second_radar_rscv': 'output/second_vod_radar/pp_radar_rcs_doppler/eval/checkpoint_epoch_80'
        'pp_lidar': 'output/pointpillar_vod_lidar/debug_new/eval/checkpoint_epoch_80',
        '3dssd_lidar': 'output/3DSSD_vod_lidar/all_cls/eval/checkpoint_epoch_80',
        'centerpoint_lidar': 'output/centerpoint_vod_lidar/xyzi/eval/best_epoch_checkpoint'
    
    }

    ## todo: centerpoint-lidar


    draw_gt = True
    
    lidar_range = [0, -25.6, -3, 51.2, 25.6, 1]

    for tag in path_dict.keys():
        is_radar = False if 'lidar' in tag.lower() else True
        modality = 'radar' if is_radar else 'lidar'
        print(f'VISUALIZING TAG: {tag}')
        result_path = base_path / path_dict[tag]
        data_path = base_path / ('data/vod_%s/training/velodyne'%modality )

        save_base_path = base_path /'output' / 'vod_vis'
        save_base_path.mkdir(exist_ok=True)
        save_path = save_base_path / tag
        save_path.mkdir(exist_ok=True)

        dt_img_path = save_path/'dt_img'
        dt_img_path.mkdir(exist_ok=True)
        # os.makedirs(dt_img_path)

        data_ids = np.loadtxt(str(result_path / 'frame_ids.txt'), delimiter=',', dtype=str)[:-1]

        with open(str(result_path / 'gt.pkl'), 'rb') as f:
            gt = pickle.load(f)

        # load det
        with open(str(result_path / 'dt.pkl'), 'rb') as f:
            dt = pickle.load(f)

        keys = list(gt.keys())
        cls_name = ['Car','Pedestrian', 'Cyclist', 'Others']
        color_dict = {}
        for i, v in enumerate(cls_name):
            color_dict[v] = label_color_palette_2d[v]

        if draw_gt:
            gt_img_path = save_path/'gt_img'
            gt_img_path.mkdir(exist_ok=True)
            saveODImgs(data_ids, gt, data_path, gt_img_path, \
                color_dict, is_radar=is_radar, title='gt', limit_range=lidar_range)
            gt_imgs = sorted(glob(str(gt_img_path/'*.png')))
            make_vid(gt_imgs, save_path/'gt.mp4', fps=10)

        saveODImgs(data_ids, dt, data_path, dt_img_path, \
        color_dict, is_radar=is_radar, title=tag, limit_range=lidar_range)

        dt_imgs = sorted(glob(str(dt_img_path/'*.png')))

        make_vid(dt_imgs, save_path/'dt.mp4', fps=10)
