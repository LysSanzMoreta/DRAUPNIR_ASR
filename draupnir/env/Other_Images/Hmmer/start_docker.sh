#!/bin/sh
#TODO: before it was bash, now sh

#!/bin/bash
if command -v "docker" &> /dev/null;then
    echo "docker found"
else
    echo "docker not found, setting alias docker to podman"
    alias docker=podman
fi


#-v /storage/Lys/draupnir_data:/opt/project/draupnir_data:Z

sudo docker run --rm -it --security-opt=label=disable -v $HOME/.bash_docker_history:/root/.bash_history  -P hmmerdkr bash