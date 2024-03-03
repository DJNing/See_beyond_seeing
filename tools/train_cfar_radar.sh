python3 tools/train.py \
--cfg_file /CenterPoint-KITTI/tools/cfgs/kitti_models/IA-SSD-GAN-vod-aug-radar.yaml \
--pretrained_model /CenterPoint-KITTI/output/IA-SSD-vod-radar/rcs_vcomp_bs64_lr0.01/eval/eval_with_train/best_eval/best_epoch_checkpoint.pth \
--batch_size 32 \
--epochs 40 \
--workers 8 \
--extra_tag cfar_radar_attach_lidar_ckpt