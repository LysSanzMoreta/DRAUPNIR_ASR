FROM docker.io/mambaorg/micromamba:1.5-jammy-cuda-12.2.2 as micromamba310
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /tmp/requirements.txt
#COPY --chown=$MAMBA_USER:$MAMBA_USER dsp-cell-annotation-service-89cb0e63bf44.json /tmp/dsp-cell-annotation-service-89cb0e63bf44.json
#COPY --chown=$MAMBA_USER:$MAMBA_USER broad-dsde-methods-7d2e953fb829.json /tmp/broad-dsde-methods-7d2e953fb829.json

USER root
RUN micromamba install -y -n base -f /tmp/env.yaml && \
micromamba clean --all --yes

#RUN micromamba install -y -n base -c conda-forge scvi-tools==1.0.4
#ENV PATH /opt/conda/bin:$PATH #retrieves pip to the environment
#RUN pip install --yes gget
ENV PATH /usr/bin/*:$PATH

RUN apt-get update

#Install R packages

##For running using GUI
RUN apt-get --yes install x11-apps && rm -rf /var/lib/apt/lists/*


RUN chmod -R 777 ./
RUN chmod -R 770 ./
RUN chown 101:101 ./ #change ownership

######################################################

CMD ["/usr/bin/which","python3"] -> "/opt/conda/bin/python3"
CMD ["/opt/conda/bin/python3","/opt/.pycharm_helpers/packaging_tool.py","list"]
CMD ["/opt/conda/bin/python3","/opt/.pycharm_helpers/remote_sync.py","--state-file","/tmp/e06641bc-ea0b-4a32-a780-2ed112e01659/.state.json","/tmp/bacecef2-e56a-4f46-9a40-fc180ff6cdd3"]

WORKDIR /opt/project
#Highlight: GUI interface activation?
#activate only for debugging GUI activation issues
#ENV QT_DEBUG_PLUGINS=1
#Mount volumes in dockerfile
VOLUME /tmp/.X11-unix:/tmp/.X11-unix
#VOLUME /run/user/1000/gdm/Xauthority:/root/.Xauthority #necessary?

#RUN xauth merge /root/.Xauthority #necessary?
ENV DISPLAY=:0

