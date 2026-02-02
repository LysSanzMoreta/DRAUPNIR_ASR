#!/bin/bash
#podman system prune -a #clean all the containers, images etc
podman container prune
podman run --rm -it --device=nvidia.com/gpu=all --security-opt=label=disable --shm-size 12G -v /home/lys/Dropbox/PhD/DRAUPNIR_ASR:/opt/project:Z -w /opt/project/ESM_experiment draupnir_xlstm2 bash