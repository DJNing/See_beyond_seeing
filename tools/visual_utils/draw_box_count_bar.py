import io as sysio
from operator import gt

import numba
import numpy as np
import pickle
from pathlib import Path as P
from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import rotate_iou_gpu_eval
from pcdet.datasets.kitti.kitti_object_eval_python.eval import clean_data,_prepare_data,eval_class,get_mAP,get_mAP_R40,calculate_iou_partly, get_split_parts, compute_statistics_jit
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
        # if not is_dt and key == '00050' and not is_radar:
        #     # debug
        #     print(points_in_box_count)
        #     print(points_in_box_count)
        #     print(points_in_box_count)
        #     # import ipdb;ipdb.set_trace()
    return pkl_file

def adjust_lightness(color, amount=0.5):
    # https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def counts_tp_dt_boxes(gt_annos, dt_annos, current_classes, difficultys, metric, min_overlaps, distance_range=None, box_count_threshold=None, num_parts=100):
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    result_dir = dict()
    for m, current_class in enumerate(current_classes):
        result_dir[current_class] = dict()
        for l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty, distance_range, box_count_threshold)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                total_tp_count = 0
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False)
                    tp, fp, fn, similarity, thresholds = rets
                    total_tp_count += tp
                result_dir[current_class]['min_overlap'] = min_overlap
                result_dir[current_class]['gt_counts'] = total_num_valid_gt
                result_dir[current_class]['dt_tp_counts'] = total_tp_count
                # import ipdb; ipdb.set_trace()
    return result_dir

def generate_range(start, end, step):
    start_range = list(range(start, end, step))
    end_range = list(range(start+step, end+step, step))

    # append infinity
    start_range += [end_range[-1]]
    end_range += [float('inf')]

    result = np.array([start_range, end_range])
    result = result.T
    # result = result.astype(int)
    return result

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
        point_counts_in_box = get_lidar_range()
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], 
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    iou_threshold = np.expand_dims(overlap_0_5,axis=0)

    cnt_str =  [f"{counts_threshold[0]}-{counts_threshold[1]}" for counts_threshold in point_counts_in_box]

    def get_cnt_str(point_counts_in_box):
        str_list = []
        for thres in point_counts_in_box:
            start_str = str(int(thres[0]))
            if thres[1] == np.inf:
                end_str = 'inf'
            else:
                end_str = str(int(thres[1]))
            cur_str = '[%s, %s)' % (start_str, end_str)
            str_list += [cur_str]
        return str_list

    str_list = get_cnt_str(point_counts_in_box)

    count_results = {
        'Car': {
            'min_overlap': [],
            'gt_counts': [],
            'dt_tp_counts': []
        },
        'Pedestrian':{
            'min_overlap': [],
            'gt_counts': [],
            'dt_tp_counts': []
        },
        'Cyclist':{
            'min_overlap': [],
            'gt_counts': [],
            'dt_tp_counts': []
        },
        'counts': str_list
    }

    for counts_threshold in point_counts_in_box:
        ret = counts_tp_dt_boxes(gt_annos, dt_annos, current_classes, difficulties, 2,
                     iou_threshold,box_count_threshold=counts_threshold)
        count_results["Car"]['min_overlap'] += [ret[0]['min_overlap']]
        count_results["Car"]['gt_counts'] += [ret[0]['gt_counts']]
        count_results["Car"]['dt_tp_counts'] += [ret[0]['dt_tp_counts']]

        count_results["Pedestrian"]['min_overlap'] += [ret[1]['min_overlap']]
        count_results["Pedestrian"]['gt_counts'] += [ret[1]['gt_counts']]
        count_results["Pedestrian"]['dt_tp_counts'] += [ret[1]['dt_tp_counts']]

        count_results["Cyclist"]['min_overlap'] += [ret[2]['min_overlap']]
        count_results["Cyclist"]['gt_counts'] += [ret[2]['gt_counts']]
        count_results["Cyclist"]['dt_tp_counts'] += [ret[2]['dt_tp_counts']]
    return count_results

def draw_results(counts_threshold,
                count_result,
                result_dir,
                color,
                class_name,
                fig_name=None,
                xlabel=None,
                fig_title=None):
    fig, ax = plt.subplots(1, figsize=(16,8))

    ax.bar(counts_threshold, count_result['dt_tp_counts'], color=color, )
    ax.bar(counts_threshold, np.array(count_result['gt_counts']) - np.array(count_result['dt_tp_counts']),bottom=count_result['dt_tp_counts'], color=adjust_lightness(color, 0.1))

    ax.set_xlabel('# of points in the boxes') 
    ax.set_ylabel('# of boxes')

    offset = int(len(ax.patches) / 2)
    labels = (np.array(count_result['dt_tp_counts']) / np.array(count_result['gt_counts'])) * 100
    # import ipdb; ipdb.set_trace()
    for i in range(len(labels)):
        height = ax.patches[i].get_height() + ax.patches[i+offset].get_height()
        rect = ax.patches[i]
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 10, '{:.2f}%'.format(labels[i]), ha="center", va="bottom"
        )

    # ax.set_yticks(np.arange(0, 110, 10))
    
    # ax.set_xticks(np.arange(0, 10, 1))
    
    ax.grid(axis = 'y')
    
    ax.legend(['Pred %s 3D IoU@%s' % (class_name, count_result['min_overlap'][0]), 'GT Boxes Counts'])

    for label in ax.get_yticklabels()[1::2]:
        label.set_visible(False)
        # plt.xlim(xmin=0) 

    # plt.ylim(ymin=0,ymax=100)

    if fig_name is not None:
        fig_path = result_dir / (fig_name+'.png')
    else:
        fig_path = result_dir / 'iou_threshold.png'

    if fig_title is not None:
        ax.set_title(fig_title)

    fig.savefig(fig_path)
    plt.close()

def draw_result_combine(count_result,
                result_dir,
                fig_name=None,
                xlabel=None,
                fig_title=None):
    fig, ax = plt.subplots(1, figsize=(28,8))
    
    n_groups = len(count_result['counts'])
    index = np.arange(n_groups)/3

    bar_width = 0.1

    ax.grid(axis = 'y')
    car_color = label_color_palette_2d['Car']
    ped_color = label_color_palette_2d['Pedestrian']
    cyclist_color = label_color_palette_2d['Cyclist']

    dt_car = ax.bar(index, count_result['Car']['dt_tp_counts'], bar_width, color=car_color)
    ax.bar(index, np.array(count_result['Car']['gt_counts']) - np.array(count_result['Car']['dt_tp_counts']),
            bar_width,bottom=count_result['Car']['dt_tp_counts'], color=adjust_lightness(car_color, 0.1))

    based = 0
    offset = int(len(ax.patches) / 2)
    labels = (np.array(count_result['Car']['dt_tp_counts']) / np.array(count_result['Car']['gt_counts'])) * 100
    for i in range(len(labels)):
        height = ax.patches[i+based].get_height() + ax.patches[i+based+offset].get_height()
        rect = ax.patches[i]
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 10, '{:.2f}%'.format(labels[i]), ha="center", va="bottom"
        )


    dt_pred = ax.bar(index+bar_width, count_result['Pedestrian']['dt_tp_counts'], bar_width, color=ped_color)
    ax.bar(index+bar_width, np.array(count_result['Pedestrian']['gt_counts']) - np.array(count_result['Pedestrian']['dt_tp_counts']),
            bar_width,bottom=count_result['Pedestrian']['dt_tp_counts'], color=adjust_lightness(car_color, 0.1))
    
    based += offset * 2
    labels = (np.array(count_result['Pedestrian']['dt_tp_counts']) / np.array(count_result['Pedestrian']['gt_counts'])) * 100
    for i in range(len(labels)):
        height = ax.patches[i+based].get_height() + ax.patches[i+based+offset].get_height()
        rect = ax.patches[i+based]
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 10, '{:.2f}%'.format(labels[i]), ha="center", va="bottom"
        )

    dt_cyclist = ax.bar(index+bar_width*2, count_result['Cyclist']['dt_tp_counts'], bar_width, color=cyclist_color)
    gt = ax.bar(index+bar_width*2, np.array(count_result['Cyclist']['gt_counts']) - np.array(count_result['Cyclist']['dt_tp_counts']),
            bar_width,bottom=count_result['Cyclist']['dt_tp_counts'], color=adjust_lightness(car_color, 0.1))
    
    based += offset * 2
    labels = (np.array(count_result['Cyclist']['dt_tp_counts']) / np.array(count_result['Cyclist']['gt_counts'])) * 100
    for i in range(len(labels)):
        height = ax.patches[i+based].get_height() + ax.patches[i+based+offset].get_height()
        rect = ax.patches[i+based]
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 10, '{:.2f}%'.format(labels[i]), ha="center", va="bottom"
        )

    ax.set_xlabel('# of points in the boxes') 
    ax.set_ylabel('# of boxes')

    
    overall_legend = []
    class_name = 'Car'
    legend = ['Pred %s 3D IoU@%s' % (class_name, count_result[class_name]['min_overlap'][0])]
    overall_legend = overall_legend + legend
    class_name = 'Pedestrian'
    legend = ['Pred %s 3D IoU@%s' % (class_name, count_result[class_name]['min_overlap'][0])]
    overall_legend = overall_legend + legend
    class_name = 'Cyclist'
    legend = ['Pred %s 3D IoU@%s' % (class_name, count_result[class_name]['min_overlap'][0]), 'GT Boxes Counts']
    overall_legend = overall_legend + legend

    ax.legend([dt_car, dt_pred, dt_cyclist, gt], overall_legend)

    ax.set_xticks(index+bar_width, count_result['counts'])
    
    bottom, top = plt.ylim()
    plt.ylim(bottom, top)
    for label in ax.get_yticklabels()[1::2]:
        label.set_visible(False)
        # plt.xlim(xmin=0) 

    # plt.ylim(ymin=0,ymax=100)

    if fig_name is not None:
        fig_path = result_dir / (fig_name+'.png')
    else:
        fig_path = result_dir / 'box_count_bar.png'

    if fig_title is not None:
        ax.set_title(fig_title)
    fig.set_figheight(6)
    fig.set_figwidth(len(count_result['counts']) * 3)
    fig.savefig(fig_path)
    plt.close()
    print('===========================================')
    print('saved drawing for %s ' % fig_path)
    print('===========================================')


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
        # tag = 'radar_rcs'
        abs_path = P(__file__).parent.resolve()
        base_path = abs_path.parents[1]
        # base_path = P('/mnt/12T/DJ/PCDet_output')
        result_path = base_path / path_dict[tag]

        modality = 'radar' if  is_radar[tag] else 'lidar'
        data_path = base_path / ('data/vod_%s/training/velodyne'%modality )

        # save_path = base_path / 'vod_vis' / 'box_count'
        save_path = base_path / 'output' / 'vod_vis' / 'box_count'
        # save_path = base_path / 'output' / 'vod_vis' / 'points_in_box_bar'
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
        count_results = get_all_box_count_result(new_gt,new_dt,current_classes, is_radar=is_radar[tag])

        print(count_results)

        # draw plots
        # draw_results(count_results['counts'],
        #         count_results['Car'],
        #         save_path,
        #         class_name = 'Car',
        #         color=label_color_palette_2d['Car'],
        #         fig_name='box_point_count_' + str(tag) + '_Car',
        #         fig_title=tag)
        
        # draw_results(count_results['counts'],
        #         count_results['Pedestrian'],
        #         save_path,
        #         class_name = 'Pedestrian',
        #         color=label_color_palette_2d['Pedestrian'],
        #         fig_name='box_point_count_' + str(tag) + '_Pedestrian',
        #         fig_title=tag)
        
        # draw_results(count_results['counts'],
        #         count_results['Cyclist'],
        #         save_path,
        #         class_name = 'Cyclist',
        #         color=label_color_palette_2d['Cyclist'],
        #         fig_name='box_point_count_' + str(tag) + '_Cyclist',
        #         fig_title=tag)
        
        # draw combine plots
        print('===========================================')
        draw_result_combine(count_results,save_path,fig_name='box_point_count_' + str(tag) + '_COMBINE',fig_title=tag)
        print('saved drawing for %s ' % tag)
        print('===========================================')
 
        # break

if __name__ == "__main__":
    main()
