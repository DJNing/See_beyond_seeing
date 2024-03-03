docker run -it --ipc=host \
    -p 8986:22 \
    -v /home/gabriel/data:/root/data \
    -v /home/gabriel/CenterPoint-KITTI:/CenterPoint-KITTI \
    --gpus all \
    --name kitti-g \
    torch:kittig