
#!/bin/bash
if command -v "docker" &> /dev/null;then
    echo "docker found"
else
    echo "docker not found, setting alias docker to podman"
    alias docker=podman
fi


docker image rm draupnir39 --force #remove image completely
#docker rm $(docker ps -a | grep -v "pycharm" | awk 'NR>1 {print $1}') #remove old containers except pycharm
docker rm -v -f $(docker ps -qa) # this line is optional !!!!!!!!!!! it will delete all existing constainers, which might be helpful to clean out everything or be annoying
docker builder prune #delete cache
docker rmi $(docker images -f "dangling=true" -q) #remove untagged/uncompleted images
docker build --no-cache --tag draupnir39 . #build new image with same name ignoring cache
#docker-compose up --build
