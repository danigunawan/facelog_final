FROM ubuntu:18.04 as nvidia_base

MAINTAINER luuvanthanh15dt2@gmail.com

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt update --fix-missing && apt install -y wget curl bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion systemd python3-dev default-libmysqlclient-dev net-tools iputils-ping vim htop && \
    rm /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

# Setup NVIDIA Base
RUN apt update && apt install -y --no-install-recommends gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV NCCL_VERSION 2.4.2
ENV CUDA_VERSION 10.0.130
ENV CUDNN_VERSION 7.5.1.10

LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

ENV CUDA_PKG_VERSION 10-0=$CUDA_VERSION-1
# For libraries in the cuda-compat-*
RUN apt update && apt install -y --no-install-recommends \
    cuda-cudart-$CUDA_PKG_VERSION \
    cuda-compat-10-0 \
    cuda-libraries-$CUDA_PKG_VERSION \
    cuda-libraries-dev-$CUDA_PKG_VERSION \
    cuda-nvml-dev-$CUDA_PKG_VERSION \
    cuda-nvtx-$CUDA_PKG_VERSION \
    cuda-minimal-build-$CUDA_PKG_VERSION \
    cuda-command-line-tools-$CUDA_PKG_VERSION \
    libcudnn7=$CUDNN_VERSION-1+cuda10.0 \
    libcudnn7-dev=$CUDNN_VERSION-1+cuda10.0 \
    libnccl2=$NCCL_VERSION-1+cuda10.0 \
    libnccl-dev=$NCCL_VERSION-1+cuda10.0 && \
    apt-mark hold libcudnn7 libcudnn7-dev && \
    apt-mark hold libnccl2 libnccl-dev && \
    ln -s cuda-10.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/cmake-3.14.3/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/opt/OpenBLAS/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
#install python3 and pip3 
RUN apt-get install python3-dev
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py
# Stage 2: Install  dependencies
FROM nvidia_base as facelog_v2

RUN pip install -i https://test.pypi.org/simple/ hung-utils==0.6.0

RUN pip install --upgrade pip


WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

CMD ["python3", "/app/mainnew.py"]


