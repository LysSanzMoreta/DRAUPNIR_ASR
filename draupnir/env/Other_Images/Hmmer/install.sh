#!/bin/bash
if command -v "docker" &> /dev/null;then
    echo "docker found"
else
    echo "docker not found, setting alias docker to podman"
    alias docker=podman
fi

name=hmmerdkr

docker image rm $name --force #remove image completely
##docker rm $(docker ps -a | grep -v "pycharm" | awk 'NR>1 {print $1}') #remove old containers except pycharm
##docker rm -v -f $(docker ps -qa) #removes containers both currently running (ps, process status) and stopped ones
#docker builder prune #delete cache
#docker container prune -a # remove all stopped and active containers
docker rmi $(docker images -f "dangling=true" -q) #remove untagged/uncompleted images
docker build --no-cache --tag $name . #build new image with same name ignoring cache
