ARG SRC_IMAGE
FROM ${SRC_IMAGE}

SHELL ["/bin/bash", "-c"]
RUN sed -i.bak -e "s%http://[^ ]\+%http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%g" /etc/apt/sources.list

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-numpy-dev \
        libpcl-dev \
        libopencv-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN if [[ ! -f /usr/lib/x86_64-linux-gnu/libboost_numpy3.so ]]; then \
        ln -s /usr/lib/x86_64-linux-gnu/libboost_numpy3*.so /usr/lib/x86_64-linux-gnu/libboost_numpy3.so ; \
    fi; \
    if [[ ! -f /usr/lib/x86_64-linux-gnu/libboost_python3.so ]]; then \
        ln -s /usr/lib/x86_64-linux-gnu/libboost_python3*.so /usr/lib/x86_64-linux-gnu/libboost_python3.so ; \
    fi

RUN if [[ -f /opt/conda/bin/activate ]]; then \
        source /opt/conda/bin/activate && \
        conda install -c conda-forge scikit-build ninja h5py==2.10.0 argcomplete optuna optuna-dashboard openpyxl pymysql; \
    else \
        pip3 install scikit-build ninja h5py==2.10.0 argcomplete optuna optuna-dashboard openpyxl pymysql; \
    fi; \
    pip install --verbose git+https://github.com/shikishima-TasakiLab/pointsmap-python && \
    pip install git+https://github.com/shikishima-TasakiLab/h5dataloader && \
    pip install git+https://github.com/shikishima-TasakiLab/h5datacreator && \
    pip install adabelief-pytorch

RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof tensorboard && \
    pip install tensorboard torch-tb-profiler==0.1.0

RUN conda uninstall PyYAML -y 
RUN pip install open3d && pip uninstall markupsafe -y && pip install markupsafe

RUN pip install optuna
RUN pip uninstall pillow -y && pip install pillow==9.5.0

RUN pip uninstall numpy -y && pip install numpy==1.20.3
RUN pip install -U tqdm && pip install opencv-python

RUN echo "export PS1='\[\033[01;32m\]pmod-net@docker\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /root/.bashrc

