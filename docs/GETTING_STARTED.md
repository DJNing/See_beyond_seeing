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






