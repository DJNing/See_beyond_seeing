python3 tools/train.py \
--cfg_file /seeing_beyond/tools/cfgs/kitti_models/cfar-lidar.yaml \
--pretrained_model /seeing_beyond/ckpts/iassd/iassd-lidar.pth \
--batch_size 28 \
--epochs 40 \
--workers 8 