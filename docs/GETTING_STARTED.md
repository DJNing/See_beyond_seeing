# Step 0: Prerequisites 
Please make sure you have [docker](https://docs.docker.com/engine/install/ubuntu/) and [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

Of course, you will need an Nvidia GPU.



# Docker
Assuming your cd is the repository root:

```
cd docker
./build.sh
```

Alternatively, an image is already built and pushed to docker hub at `https://hub.docker.com/r/gc625kodifly/seeing_beyond`. Please run 
```
docker pull docker.io/gc625kodifly/seeing_beyond:latest
```

# Dataset 

## Download 
Next, please request access and download the view of delft dataset:
```
https://tudelft-iv.github.io/view-of-delft-dataset/#access
```

next, please extract the dataset to desired location. The directory structure should look something like this:

```
/path/to/where/you/store/data
└── view_of_delft_PUBLIC
    ├── lidar
    │   ├── gt_database
    │   ├── ImageSets
    │   ├── testing
    │   └── training
    ├── radar
    │   ├── gt_database
    │   ├── ImageSets -> ../lidar/ImageSets/
    │   ├── testing
    │   └── training
    ├── radar_3frames
    │   ├── ImageSets -> ../lidar/ImageSets/
    │   ├── testing
    │   └── training
    └── radar_5frames
        ├── ImageSets -> ../lidar/ImageSets/
        ├── testing
        └── training
```

# Running Docker 

Next, please run 

```
./docker/run.sh <repository-path> <data-path>

# e.g.
# ./docker/run.sh /home/user/See_beyond_seeing /home/user/data
# where /home/user/data contains view_of_delft_PUBLIC folder
```

# Preprocessing Dataset
once you are inside the container, please run:

```
cd /seeing_beyond
python3 /seeing_beyond/tools/prepare_vod_dataset.py

python3 setup.py develop 

cd /seeing_beyond/pcdet/ops/pointnet2/pointnet2_3DSSD
python3 setup.py develop

python3 -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos /seeing_beyond/tools/cfgs/dataset_configs/vod_lidar_dataset.yaml
python3 -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos /seeing_beyond/tools/cfgs/dataset_configs/vod_radar_dataset.yaml
```


# Evaluation

we provide pretrained models at under `./ckpts`. To run eval on the checkpoints, please run the following scripts inside the container:
```
# LiDAR main modality, radar aux
./seeing_beyond/tools/eval_cfar_lidar.sh
# Radar main modality, LiDAR aux
/seeing_beyond/tools/eval_cfar_radar.sh
``` 

# Training

Training scripts are also provided, please run:
```
# LiDAR main modality, radar aux
./seeing_beyond/tools/trian_cfar_lidar.sh
# Radar main modality, LiDAR aux
/seeing_beyond/tools/trian_cfar_radar.sh
```





