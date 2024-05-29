#!/bin/bash

# Function to display the help message
show_help() {
  echo "Usage: $0 <repository-path> <data-path>"
  echo
  echo "Arguments:"
  echo "  <repository-path>    Path to the where https://github.com/DJNing/See_beyond_seeing is cloned."
  echo "  <data-path>          Path to the data directory where view_of_delft_PUBLIC is located."
  echo
  echo "Example:"
  echo "  $0 /path/to/repository /path/to/data"
}

# Check if help is requested
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
  show_help
  exit 0
fi

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
  show_help
  exit 1
fi

# Assign the arguments to variables
REPOSITORY_PATH=$1
DATA_PATH=$2

# Run the Docker container with the provided paths
docker run -it --ipc=host --network host \
    -v ${DATA_PATH}:/root/data \
    -v ${REPOSITORY_PATH}:/seeing_beyond \
    --gpus all \
    --name seeing_beyond \
    docker.io/gc625kodifly/seeing_beyond:latest \
    /bin/bash
