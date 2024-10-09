from turtle import color
import numpy as np
import pickle
from pathlib import Path as P
from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import rotate_iou_gpu_eval
from pcdet.datasets.kitti.kitti_object_eval_python.eval import clean_data,_prepare_data,eval_class,get_mAP,get_mAP_R40
from pcdet.datasets.kitti.kitti_object_eval_python.kitti_common import get_label_annos
from vod.visualization.settings import label_color_palette_2d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



class graph_stuff:

    def __init__(self) -> None:
        self.linestyle_tuple = [
            ('solid', 'solid'),
            ('dotted',                (0, (1, 1))),
            ('loosely dashed',        (0, (5, 10))),
            ('dashdotted',            (0, (3, 5, 1, 5))),
            ('densely dashdotted',    (0, (3, 1, 1, 1))),
            ('loosely dotted',        (0, (1, 10))),
            ('loosely dashdotted',    (0, (3, 10, 1, 10))),
            ('dashed',                (0, (5, 5))),
            ('densely dotted',        (0, (1, 1))),
            ('long dash with offset', (5, (10, 3))),
            ('densely dashed',        (0, (5, 1))),
            ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
            ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
            ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
        self.markers = list(Line2D.markers)


def draw_one(x,car,ped,cyclist,tag,ax,linestyle,marker,draw_car,
    draw_ped,
    draw_cyclist,
    draw_mAP,
    color_modifier,
    size):
    car_color = list(label_color_palette_2d['Car'])
    ped_color = list(label_color_palette_2d['Pedestrian'])
    cyclist_color = list(label_color_palette_2d['Cyclist'])
    mAP_color = [0,0,0]
    mAP = np.mean([car,ped,cyclist],axis=0)
    

    if draw_car:
        car_colors = np.linspace(car_color[2],0.1,size)
        car_color[2] = car_colors[color_modifier]

        ax.plot(
            x,
            car,
            color=car_color,
            label=f'{tag}',
            linestyle=linestyle,
            marker=marker)
    
    if draw_ped:
        ped_colors = np.linspace(ped_color[1],1,size)
        ped_color[1] = ped_colors[color_modifier]
        ax.plot(
            x,
            ped,
            color=ped_color,
            label=f'{tag}',
            linestyle=linestyle,
            marker=marker)

    if draw_cyclist:
        # cyclist_color[0] -= color_modifier
        cyclist_colors = np.linspace(cyclist_color[0],0.1,size)
        cyclist_color[0] = cyclist_colors[color_modifier]
        ax.plot(
            x,
            cyclist,
            color=cyclist_color,
            label=f'{tag}',
            linestyle=linestyle,
            marker=marker)
    if draw_mAP:
        blacks = np.linspace(0,1,size)
        mAP_color[0] = blacks[color_modifier]
        mAP_color[1] = blacks[color_modifier]
        mAP_color[2] = blacks[color_modifier]
        ax.plot(x,mAP,color=mAP_color,label=f'{tag}',linestyle=linestyle,marker=marker)


def compare_multiple(
    x,
    list_of_results,
    list_of_tags,
    result_dir,
    draw_car,
    draw_ped,
    draw_cyclist,
    draw_mAP,
    xlabel):

    # list of linestyles and markers for plotting    
    linestyle_tuple = graph_stuff().linestyle_tuple
    markers = graph_stuff().markers


    fig, ax = plt.subplots(1)
    color_modifier = 0
    for i,result in enumerate(list_of_results):
        current_tag = list_of_tags[i]        
        car= result['Car']['mAP_3d']
        pedestrian = result['Pedestrian']['mAP_3d']
        cyclist= result['Cyclist']['mAP_3d']        

        # FOR EACH TAG, DRAW THE GRAPH
        draw_one(
            x,
            car,
            pedestrian,
            cyclist,
            current_tag,
            ax,
            linestyle=linestyle_tuple[i][1],
            marker = markers[i],
            draw_car = draw_car,
            draw_ped = draw_ped,
            draw_cyclist = draw_cyclist,
            draw_mAP = draw_mAP,
            color_modifier = color_modifier,
            size=len(list_of_tags))

        color_modifier += 1
        
    
    ax.grid(axis = 'y')
    plt.ylim(ymin=0,ymax=100)
    
    ax.set_xlabel(xlabel) 
    ax.set_ylabel('AP (3D bounding box)')
    
    ax.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 - box.height * 0.01,
                 box.width, box.height * 0.9])

    handles, labels = plt.gca().get_legend_handles_labels()
    
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    # ax.legend(handles, labels)                 
    ax.legend(handles, labels, bbox_to_anchor=(0.5, 1.10),loc='lower center',
          fancybox=True, shadow=True, ncol=len(list_of_tags), prop={'size': 8})


    file_str =  "("
    file_str += "Car" if draw_car else ""
    file_str += "Ped" if draw_ped else ""
    file_str += "Cyclist" if draw_cyclist else ""
    file_str += "mAP" if draw_mAP else ""
    file_str += ")"
    
    title_str = ""
    title_str += "Car " if draw_car else ""
    title_str += "Ped " if draw_ped else ""
    title_str += "Cyclist " if draw_cyclist else ""
    title_str += "mAP" if draw_mAP else ""

    ax.set_title(f"Comparing {title_str}")
    fig_path = result_dir / (f'{list_of_tags[0]}_and_{len(list_of_tags)-1}others{file_str}'+'.png')
    fig.savefig(fig_path)


def load_results(list_of_tags,base_path,path_dict):

    list_iou_results = []
    list_distance_results = []

    for tag in list_of_tags:
        result_path = base_path / path_dict[tag]
        with open(result_path / 'all_iou_results.pkl', 'rb') as f:
            iou_results = pickle.load(f)
        with open(result_path / 'distance_results.pkl', 'rb') as f:
            distance_results = pickle.load(f)

        list_iou_results += [iou_results]
        list_distance_results += [distance_results]

    return list_iou_results,list_distance_results


def plot_all(iou_thresholds,
        list_of_results,
        list_of_tags,
        result_dir,
        xlabel):
        
        for i in range(4):
            flag = [0,0,0,0]
            flag[i] = 1

            compare_multiple(
            iou_thresholds,
            list_of_results=list_of_results,
            list_of_tags=list_of_tags,
            result_dir=result_dir,
            draw_car = flag[0],
            draw_ped = flag[1],
            draw_cyclist = flag[2],
            draw_mAP = flag[3],
            xlabel = xlabel)  

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

    # CHOSE THE RESULTS HERE:
    # put the main result at index=0
    list_of_tags = ['CFAR_radar','radar_rcsv','radar_rcs','radar_v','radar']
    # list_of_tags = ['CFAR_radar','radar_rcsv']

    # path stuff
    abs_path = P(__file__).parent.resolve()
    base_path = abs_path.parents[1]
    save_path = base_path /'output' / 'vod_vis' / 'comparisons'

    # load in the AP@diff_iou, AP@different_distance_ranges
    list_iou_results,list_distance_results = load_results(list_of_tags,base_path,path_dict)
    
    # get the x_axis labels for each type of results
    distance_range = list_distance_results[0]['distances']
    iou_thresholds = list_iou_results[0]['iou_thresholds']

    compare_multiple(
        iou_thresholds,
        list_of_results=list_iou_results,
        list_of_tags=list_of_tags,
        result_dir=save_path,
        draw_car = True,
        draw_ped = True,
        draw_cyclist = False,
        draw_mAP = False,
        xlabel = 'Distance from ego-vehicle')   


    plot_all(
        iou_thresholds,
        list_of_results=list_iou_results,
        list_of_tags=list_of_tags,
        result_dir=save_path,
        xlabel = 'Distance from ego-vehicle')       
    
    


if __name__ == "__main__":
    main()




