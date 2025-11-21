#!/bin/bash
#podman system prune -a #clean all the containers, images etc
podman container prune
podman run --rm -it --gpus 1 --security-opt=label=disable --device=nvidia.com/gpu=all --shm-size 12G -v /home/lys/Dropbox/PhD/DRAUPNIR_ASR:/opt/project:Z -w /opt/project/ESM_experiment draupnirdocker bash