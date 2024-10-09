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
    'pp_radar_rcs' : 'output/pointpillar_vod_radar/debug_new/eval/checkpoint_epoch_80',
    'pp_radar_rcsv' : 'output/pointpillar_vod_radar/vrcomp/eval/best_epoch_checkpoint', 
    '3dssd_radar_rcs': 'output/3DSSD_vod_radar/rcs/eval/best_epoch_checkpoint',
    '3dssd_radar_rcsv': 'output/3DSSD_vod_radar/vcomp/eval/best_epoch_checkpoint',
    'centerpoint_radar_rcs': 'output/centerpoint_vod_radar/rcs/eval/best_epoch_checkpoint',
    'centerpoint_radar_rcsv': 'output/centerpoint_vod_radar/rcsv/eval/best_epoch_checkpoint',
    'second_radar_rcs': 'output/second_vod_radar/radar_second_with_aug/eval/checkpoint_epoch_80',
    'second_radar_rscv': 'output/second_vod_radar/pp_radar_rcs_doppler/eval/checkpoint_epoch_80',
    'second_lidar': 'output/second_vod_lidar/new_train/eval/best_ckpt',
    'pp_lidar': 'output/pointpillar_vod_lidar/debug_new/eval/checkpoint_epoch_80',
    '3dssd_lidar': 'output/3DSSD_vod_lidar/all_cls/eval/checkpoint_epoch_80',
    'centerpoint_lidar': 'output/centerpoint_vod_lidar/xyzi/eval/best_epoch_checkpoint',
    # ! CHECK PVRCNN
    'pvrcnn_lidar': '/root/gabriel/code/parent/CenterPoint-KITTI/output/pv_rcnn_vod_lidar/debug_init/eval/best_ckpt', 
    'pvrcnn_radar': '',
    'pointrcnn_lidar': '/root/gabriel/code/parent/CenterPoint-KITTI/output/pointrcnn_vod_lidar/init/eval/best_ckpt',
    # ! PointRCNN radar empty? 
    'pointrcnn_radar':'',
    'lidar_GT':'',
    'radar_GT':''
}


lidar_path_dict = {v:path_dict[v] for v in path_dict if 'lidar' in v}
radar_path_dict = {v:path_dict[v] for v in path_dict if 'radar' in v}



det_path_dict = {
'CFAR_radar':  '',
'radar_rcsv':'',
'radar_rcs':'',
'radar_v':'',
'radar':'',
'lidar_i':'output/vod_vis/vis_video/lidar_i480v6/LidarPred',
'lidar':'',
'CFAR_lidar_rcsv': '',
'CFAR_lidar_rcs': 'output/vod_vis/vis_video/CFAR_lidar_rcs480v6/LidarPred',
'CFAR_lidar_v':'',
'CFAR_lidar':'',
'pp_radar_rcs':'',
'pp_radar_rcsv': '',
'3dssd_radar_rcs':'',
'3dssd_radar_rcsv':'',
'centerpoint_radar_rcs':'',
'centerpoint_radar_rcsv':'',
'second_radar_rcs':'',
'second_radar_rscv':'',
'pp_lidar':'output/vod_vis/vis_video/pp_lidar480v6/LidarPred',
'3dssd_lidar':'output/vod_vis/vis_video/3dssd_lidar480v6/LidarPred',
'centerpoint_lidar': 'output/vod_vis/vis_video/centerpoint_lidar480v6/LidarPred',
'second_lidar':'output/vod_vis/vis_video/second_lidar480v6/LidarPred',
'pointrcnn_lidar':'output/vod_vis/vis_video/pointrcnn_lidar480v6/LidarPred',
'pvrcnn_lidar':'output/vod_vis/vis_video/pvrcnn_lidar480v6/LidarPred',
'lidar_GT':'output/vod_vis/vis_video/CFAR_lidar_rcs480v6/LidarGT',
'radar_GT': ''
}



radar_baseline_tags = [
'radar_rcsv'
'second_radar_rscv',
'pp_radar_rcsv',
'3dssd_radar_rcsv',
'centerpoint_radar_rcsv'
] 

lidar_baseline_tags = [
    'lidar_i', #
    'second_lidar', #
    'pp_lidar', # 
    '3dssd_lidar', #
    'centerpoint_lidar',
    'pvrcnn_lidar',
    'pointrcnn_lidar'
    ]