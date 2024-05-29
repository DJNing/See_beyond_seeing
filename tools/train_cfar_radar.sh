python3 tools/train.py \
--cfg_file /seeing_beyond/tools/cfgs/kitti_models/cfar-radar.yaml \
--pretrained_model /seeing_beyond/ckpts/iassd/iassd-radar-vcomp.pth \
--batch_size 32 \
--epochs 40 \
--workers 8 