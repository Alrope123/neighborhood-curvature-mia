FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
	    unzip \
	    curl \
        cmake \
        ca-certificates \
        python3-setuptools \
        python3.9 \
        python3-pip && \
     rm -rf /var/lib/apt/lists/*

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH /opt/conda/bin:/usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN pip3 install wheel
RUN python3 -m pip install -U --force-reinstall pip
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision datasets tqdm numpy matplotlib scikit-learn transformers

# Install Python dependencies
COPY MIA/ MIA/
