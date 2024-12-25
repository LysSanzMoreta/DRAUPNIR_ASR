podman image rm draupnirdocker --force #remove image completely
#podman rm $(podman ps -a | grep -v "pycharm" | awk 'NR>1 {print $1}') #remove old containers except pycharm
podman rm -v -f $(podman ps -qa) # this line is optional !!!!!!!!!!! it will delete all existing constainers, which might be helpful to clean out everything or be annoying
podman builder prune #delete cache
podman rmi $(podman images -f "dangling=true" -q) #remove untagged/uncompleted images
podman build --no-cache --tag draupnirdocker . #build new image with same name ignoring cache
#podman-compose up --build
