import numpy as np
import vis_tools
from pathlib import Path as P
from glob import glob
from vod.visualization.settings import label_color_palette_2d
if __name__ == '__main__':
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
        'CFAR_lidar':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_xyz_only/eval/best_epoch_checkpoint'
    }


    lidar_range = [0, -25.6, -3, 51.2, 25.6, 2]
    abs_path = P(__file__).parent.resolve()
    base_path = abs_path.parents[1]

    tag = 'CFAR_lidar_rcsv'

    vis_root_path = base_path / path_dict[tag]
    test_result_path = vis_root_path / 'final_result' / 'data'
    # test_result_path = P('/root/dj/code/CenterPoint-KITTI/output/root/gabriel/code/parent/CenterPoint-KITTI/RESULTS/tr19')
    test_result_files = sorted(glob(str(test_result_path / '*.txt')))
    is_radar = False if 'lidar' in tag.lower() else True
    # is_radar = False
    is_test = True
    modality = 'radar' if is_radar else 'lidar'
    split = 'testing' if is_test else 'training'
    pcd_file_path = base_path / ('data/vod_%s/%s/velodyne' % (modality, split))
    # pcd_files =

    def get_frame_id(fname):
        fname_p = P(fname)
        frame_id = str(fname_p.name).split('.')[0]
        return frame_id

    def collect_result_dict(result_files):
        result_dict = {}
        for fname in result_files:
            frame_id = get_frame_id(fname)
            annas = vis_tools.load_result_file(fname)
            result_dict[frame_id] = annas
        return result_dict
    
    annas_dict = collect_result_dict(test_result_files)

    frame_ids = sorted(list(annas_dict.keys()))

    cls_name = ['Car','Pedestrian', 'Cyclist', 'Others']
    color_dict = {}
    for i, v in enumerate(cls_name):
        color_dict[v] = label_color_palette_2d[v]

    vis_img_path = base_path / 'output' / 'vod_testset' / tag
    vis_img_path.mkdir(exist_ok=True, parents=True)

    img_title = 'lidar' if 'lidar' in tag else 'radar'
    vid_path = base_path/'output'
    vid_path.mkdir(exist_ok=True)
    vid_fname = vid_path/('%s.mp4'%tag)
    if not vid_fname.exists():
        print('saving detection BEV video...')
        vis_tools.saveODImgs(frame_ids, annas_dict, pcd_file_path, vis_img_path, color_dict,\
            is_radar, title=img_title, limit_range=lidar_range, is_test=True)
        
        # vis_tools.saveODImgs_multithread(frame_ids, annas_dict, pcd_file_path, vis_img_path, color_dict,\
        #     is_radar, title='test', limit_range=lidar_range, is_test=True)
        dt_imgs = sorted(glob(str(vis_img_path/'*.png')))
        
        vis_tools.make_vid(dt_imgs, vid_fname, fps=10)

    # save rgb
    rgb_vid = vid_path/('rgb.mp4')
    if rgb_vid.exists():
        pass
    else:
        print('saving rgb video...')
        img_files = [vis_tools.get_img_file(id, is_radar=is_radar, is_test=is_test) for id in frame_ids]
        vis_tools.make_vid(img_files, rgb_vid, fps=10)
