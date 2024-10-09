import io as sysio
from operator import gt

import numba
import numpy as np
import pickle
from pathlib import Path as P
from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import rotate_iou_gpu_eval
from pcdet.datasets.kitti.kitti_object_eval_python.eval import clean_data,_prepare_data,eval_class,get_mAP,get_mAP_R40
from pcdet.datasets.kitti.kitti_object_eval_python.kitti_common import get_label_annos
from vod.visualization.settings import label_color_palette_2d
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import matplotlib.pyplot as plt
from visualize_point_based import transform_anno, drawBEV

def get_radar(fname):
    assert fname.exists()
    radar_point_cloud = np.fromfile(str(fname), dtype=np.float32).reshape(-1, 7)
    return radar_point_cloud

def get_lidar(fname):
    assert fname.exists()
    radar_point_cloud = np.fromfile(str(fname), dtype=np.float32).reshape(-1, 4)
    return radar_point_cloud

def pcd_formating(pcd):
    num_pts = pcd.shape[0]
    zeros_pad = np.zeros([num_pts, 1])
    final_pcd = np.concatenate((zeros_pad, pcd), axis=1)
    return final_pcd

def get_rotation(yaw):
    # x,y,_ = arr[:3]
    # yaw = np.arctan(y/x)
    angle = np.array([0, 0, yaw])
    r = R.from_euler('XYZ', angle)
    return r.as_matrix()

def get_bbx_param(obj_info):

    center = obj_info[:3]
    extent = obj_info[3:6]
    angle = -(obj_info[6] + np.pi / 2)
    center[-1] += 0.5 * extent[-1]

    rot_m = get_rotation(angle)

    obbx = o3d.geometry.OrientedBoundingBox(center, rot_m, extent)
    return obbx

def count_points_in_box(pkl_file, is_radar, is_dt,data_path):
    for key in pkl_file.keys():
        if is_dt:
            anno = pkl_file[key][0]
        else:
            anno = pkl_file[key]
        loc = anno['location']
        yaw = anno['rotation_y']
        extent = anno['dimensions']
        if is_radar:
            pc = get_radar(data_path / (key + '.bin'))
        else:
            pc = get_lidar(data_path / (key + '.bin'))

        pc = pcd_formating(pc)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[:,1:4])
        points_in_box_count = []
        for cur_label_idx in range(len(anno['name'])):
            x, y, z = transform_anno(loc[cur_label_idx], key, is_radar=is_radar)
            dx, dz, dy = extent[cur_label_idx] # l, h ,w
            rot_y = yaw[cur_label_idx]
            obj_info = np.array([x, y, z, dx, dy, dz, rot_y])
            box = get_bbx_param(obj_info)
            ctr_idx = box.get_point_indices_within_bounding_box(pcd.points)
            points_in_box_count.append(len(ctr_idx))
        anno['points_in_box_count'] = points_in_box_count
    return pkl_file

def get_radar_range():
    start_range = [0, 1, 3, 6, 10]
    end_range = [1, 3, 6, 10, float('inf')]
    result = np.array([start_range, end_range])
    # result = result.astype(int)
    return result.T


def get_lidar_range():
    start_range = [0, 1, 20, 60, 120, 200]
    end_range = [1, 20, 60, 120, 200, float('inf')]
    result = np.array([start_range, end_range])
    return result.T

def get_all_box_count_result(gt_annos, dt_annos, current_classes, is_radar):
    difficulties = [0]
    if is_radar:
        point_counts_in_box = get_radar_range()
    else:
        # point_counts_in_box = [[0,100], [100,200], [200,300], [300,400], [400,500], [500,600],[600, 700],[700, 800],[800, 900]]
        point_counts_in_box = get_lidar_range()
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], 
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    iou_threshold = np.expand_dims(overlap_0_5,axis=0)

    count_results = {
        'Car': {
            'mAP_3d': [],
            'mAP_3d_R40': []
        },
        'Pedestrian':{
            'mAP_3d': [],
            'mAP_3d_R40': []
        },
        'Cyclist':{
            'mAP_3d': [],
            'mAP_3d_R40': []
        },
        'counts': [f"{counts_threshold[0]}-{counts_threshold[1]}" for counts_threshold in point_counts_in_box]
    }

    for counts_threshold in point_counts_in_box:
        ret = eval_class(gt_annos, dt_annos, current_classes, difficulties, 2,
                     iou_threshold,box_count_threshold=counts_threshold)

        mAP_3d = np.round(get_mAP(ret["precision"]),4)
        mAP_3d_R40 = np.round(get_mAP_R40(ret["precision"]),4)

        count_results["Car"]['mAP_3d'] += [mAP_3d[0].item()]
        count_results["Car"]['mAP_3d_R40'] += [mAP_3d_R40[0].item()]
                
        count_results["Pedestrian"]['mAP_3d'] += [mAP_3d[1].item()]
        count_results["Pedestrian"]['mAP_3d_R40'] += [mAP_3d_R40[1].item()]
        
        count_results["Cyclist"]['mAP_3d'] += [mAP_3d[2].item()]
        count_results["Cyclist"]['mAP_3d_R40'] += [mAP_3d_R40[2].item()]

    
    return count_results

def draw_results(counts_threshold,
                car_AP,
                pedestrian_AP,
                cyclist_AP,
                result_dir,
                is_distance=False,
                fig_name=None,
                xlabel=None,
                fig_title=None):
    fig, ax = plt.subplots(1)

    car_color = label_color_palette_2d['Car']
    ped_color = label_color_palette_2d['Pedestrian']
    cyclist_color = label_color_palette_2d['Cyclist']

    mAP = np.mean([car_AP,pedestrian_AP,cyclist_AP],axis=0)
    
    ax.scatter(counts_threshold,car_AP,color=car_color,clip_on=False)
    ax.scatter(counts_threshold,pedestrian_AP,color=ped_color,clip_on=False)
    ax.scatter(counts_threshold,cyclist_AP,color=cyclist_color,clip_on=False)
    ax.scatter(counts_threshold,mAP,color='black',clip_on=False)
    
    ax.plot(counts_threshold,car_AP,color=car_color,label='Car')
    ax.plot(counts_threshold,pedestrian_AP,color=ped_color,label='Pedestrian')
    ax.plot(counts_threshold,cyclist_AP,color=cyclist_color,label='Cyclist')
    ax.plot(counts_threshold,mAP,color='black',label='mAP')

    ax.set_xlabel('Points in Box (3D)') 
    ax.set_ylabel('AP (3D IoU)')

    ax.set_yticks(np.arange(0, 110, 10))
    
    # ax.set_xticks(np.arange(0, 10, 1))
    
    ax.grid(axis = 'y')
    
    ax.legend()

    for label in ax.get_yticklabels()[1::2]:
        label.set_visible(False)
        plt.xlim(xmin=0) 

    plt.ylim(ymin=0,ymax=100)

    if fig_name is not None:
        fig_path = result_dir / (fig_name+'.png')
    else:
        fig_path = result_dir / 'iou_threshold.png'

    if fig_title is not None:
        ax.set_title(fig_title)

    fig.savefig(fig_path)
    plt.close()



def main():
    path_dict = {
        'CFAR_radar':'output/IA-SSD-GAN-vod-aug/radar48001_512all/eval/best_epoch_checkpoint',
        'radar_rcsv':'output/IA-SSD-vod-radar/iassd_best_aug_new/eval/best_epoch_checkpoint',
        'radar_rcs':'output/IA-SSD-vod-radar/iassd_rcs/eval/best_epoch_checkpoint',
        'radar_v':'output/IA-SSD-vod-radar/iassd_vcomp_only/eval/best_epoch_checkpoint',
        # 'radar':'output/IA-SSD-vod-radar-block-feature/only_xyz/eval/best_epoch_checkpoint',
        'lidar_i':'output/IA-SSD-vod-lidar/all_cls/eval/checkpoint_epoch_80',
        # 'lidar':'output/IA-SSD-vod-lidar-block-feature/only_xyz/eval/best_epoch_checkpoint',
        'CFAR_lidar_rcsv':'output/IA-SSD-GAN-vod-aug-lidar/to_lidar_5_feat/eval/best_epoch_checkpoint',
        'CFAR_lidar_rcs':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_rcs_only/eval/best_epoch_checkpoint',
        'CFAR_lidar_v':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_vcomp_only/eval/best_epoch_checkpoint',
        'CFAR_lidar':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_xyz_only/eval/best_epoch_checkpoint',
        # 'pp_radar_rcs' : 'output/pointpillar_vod_radar/debug_new/eval/checkpoint_epoch_80',
        'pp_radar_rcsv' : 'output/pointpillar_vod_radar/vrcomp/eval/best_epoch_checkpoint', 
        # '3dssd_radar_rcs': 'output/3DSSD_vod_radar/rcs/eval/best_epoch_checkpoint',
        '3dssd_radar_rcsv': 'output/3DSSD_vod_radar/vcomp/eval/best_epoch_checkpoint',
        # 'centerpoint_radar_rcs': 'output/centerpoint_vod_radar/rcs/eval/best_epoch_checkpoint',
        'centerpoint_radar_rcsv': 'output/centerpoint_vod_radar/rcsv/eval/best_epoch_checkpoint',
        'pp_lidar': 'output/pointpillar_vod_lidar/debug_new/eval/checkpoint_epoch_80',
        '3dssd_lidar': 'output/3DSSD_vod_lidar/all_cls/eval/checkpoint_epoch_80',
    }
    is_radar = {
        'CFAR_radar': True,
        'radar_rcsv': True,
        'radar_rcs': True,
        'radar_v': True,
        'radar': True,
        'lidar_i': False,
        'lidar': False,
        'CFAR_lidar_rcsv': False,
        'CFAR_lidar_rcs': False,
        'CFAR_lidar_v': False,
        'CFAR_lidar': False,
        'pp_radar_rcsv': True,
        '3dssd_radar_rcsv':True,
        'centerpoint_radar_rcsv': True,
        'pp_lidar': False,
        '3dssd_lidar':False
    }

    for tag in path_dict.keys():
        abs_path = P(__file__).parent.resolve()
        base_path = abs_path.parents[1]
        # base_path = P('/mnt/12T/DJ/PCDet_output')
        result_path = base_path / path_dict[tag]

        modality = 'radar' if  is_radar[tag] else 'lidar'
        data_path = base_path / ('data/vod_%s/training/velodyne'%modality )

        # save_path = base_path /'temp'
        save_path = base_path /'output' / 'vod_vis' / 'box_count'
        save_path.mkdir(parents=True,exist_ok=True)

        print(f'*************   DRAWING PLOTS FOR TAG:{path_dict[tag]}   *************')

        with open(str(result_path / 'gt.pkl'), 'rb') as f:
            gt = pickle.load(f)

        with open(str(result_path / 'dt.pkl'), 'rb') as f:
            dt = pickle.load(f)

        # Car, ped, cyclis
        current_classes = [0,1,2]
        
        # import ipdb; ipdb.set_trace()

        gt = count_points_in_box(gt, is_radar[tag], is_dt=False,data_path=data_path)
        dt = count_points_in_box(dt, is_radar[tag], is_dt=True,data_path=data_path)

        # load gt boxes
        new_gt = []
        for key in gt.keys():
            new_gt += [gt[key]] 

        # load predicted boxes 
        new_dt = []
        for key in dt.keys():
            new_dt += [dt[key][0]] 

        # C
        distance_results = get_all_box_count_result(new_gt,new_dt,current_classes, is_radar=is_radar[tag])

        print(distance_results)

        # draw plots
        draw_results(distance_results['counts'],
                        distance_results['Car']['mAP_3d'],
                        distance_results['Pedestrian']['mAP_3d'],
                        distance_results['Cyclist']['mAP_3d'],
                        save_path,
                        fig_name='box_point_count_mAP_' + str(tag),
                        fig_title=tag)

        # draw_iou_results(distance_results['distances'],
        #                 distance_results['Car']['mAP_3d'],
        #                 distance_results['Pedestrian']['mAP_3d'],
        #                 distance_results['Cyclist']['mAP_3d'],
        #                 result_path,
        #                 is_distance=True,
        #                 fig_name='distances',
        #                 xlabel="Distance from ego-vehicle",
        #                 fig_title=tag)

        # print out the stuff 
        # print_results(all_iou_results,distance_results)

        # save_stats = True

        # if save_stats:
        #     with open(result_path / 'all_iou_results.pkl', 'wb') as f:
        #         pickle.dump(all_iou_results, f)

        #     with open(result_path / 'distance_results.pkl', 'wb') as f:
        #         pickle.dump(distance_results, f)    


if __name__ == "__main__":
    main()
