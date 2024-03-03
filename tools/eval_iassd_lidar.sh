export CUDA_VISIBLE_DEVICES=0 
python3 tools/eval_single_epoch.py \
--cfg_file /CenterPoint-KITTI/tools/cfgs/kitti_models/IA-SSD-vod-lidar.yaml \
--ckpt /CenterPoint-KITTI/ckpts/iassd-lidar-init_new.pt \
--pretrained_model /CenterPoint-KITTI/ckpts/iassd-lidar-init_new.pt
