FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
	    unzip \
	    curl \
        cmake \
        ca-certificates && \
     rm -rf /var/lib/apt/lists/*

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH /opt/conda/bin:/usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility


# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# RUN pip3 install wheel
# RUN python3 -m pip install -U --force-reinstall pip
# RUN pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# RUN pip3 install --upgrade pip

# Install Python dependencies
# COPY requirements.txt .
# RUN pip3 install -r requirements.txt
COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "generation", "/bin/bash", "-c"]

COPY MIA/ MIA/