FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#FROM  nvidia/cuda:9.0-base-ubuntu16.04
#FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
# FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04
# FROM  nvidia/cuda:10.1-devel-ubuntu16.04
# FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

RUN set -ex \
    && apt-get update -yqq \
    && apt-get upgrade -yqq \
    && apt-get install -yqq --no-install-recommends \
        git wget curl ssh libxrender1 libxext6 software-properties-common apt-utils libsm6 \
    && wget --no-check-certificate https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh \
    && /bin/bash Miniconda3-4.7.12.1-Linux-x86_64.sh -f -b -p /opt/miniconda \
    && add-apt-repository ppa:git-core/ppa \
    && (curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash) \
    && apt-get install git-lfs \
    && git lfs install \
    && apt-get clean \
    && /opt/miniconda/bin/conda install conda=4.8.1=py37_4 \
    && /opt/miniconda/bin/conda clean -yq -a \
    && rm Miniconda3-4.7.12.1-Linux-x86_64.sh \
    && rm -rf \
        /tmp/* \
        /var/tmp/* \
        /usr/share/man \
        /usr/share/doc \
        /usr/share/doc-base

ENV PATH /opt/miniconda/bin:$PATH

RUN  apt-get update -yqq &&  apt-get install -yqq libgl1-mesa-dev

RUN pip install jupyter ipdb -i  https://mirrors.aliyun.com/pypi/simple/

RUN  echo "10.16.94.96 github.com" >> /etc/hosts
RUN  echo "10.16.91.63 github.global.ssl.fastly.Net" >> /etc/hosts

RUN mkdir /app
COPY ./requirements.txt /app

RUN pip install -r /app/requirements.txt -i  https://mirrors.aliyun.com/pypi/simple/

WORKDIR /app
RUN wget https://anaconda.org/conda-forge/cairo/1.16.0/download/linux-64/cairo-1.16.0-hfb77d84_1002.tar.bz2
RUN wget https://anaconda.org/conda-forge/opencv/4.2.0/download/linux-64/opencv-4.2.0-py38_1.tar.bz2

RUN conda install --offline /app/cairo-1.16.0-hfb77d84_1002.tar.bz2
RUN conda install --offline /app/opencv-4.2.0-py38_1.tar.bz2
RUN rm *.bz2


RUN conda install  -c conda-forge -y pytorch==1.3.1

#RUN conda install -c conda-forge -y cairo==1.16.0
#RUN conda install  -c conda-forge -y opencv==4.2.0
#RUN conda install  -c conda-forge -y pytorch==1.3.1

RUN conda install  -c conda-forge -y torchvision==0.4.2
RUN conda install  -c conda-forge -y scipy==1.3.1
RUN conda install  -c conda-forge -y pydicom==1.3.0
RUN conda install  -c conda-forge -y matplotlib==3.1.3
RUN conda install  -c conda-forge -y scikit-image==0.15.0
RUN conda install  -c conda-forge -y Pillow==6.2.0
RUN conda install  -c conda-forge -y h5py==2.10.0
RUN conda install  -c conda-forge -y pandas==0.25.1
RUN conda install  -c conda-forge -y numpy




USER root



EXPOSE 8888
#ENTRYPOINT ["jupyter", "notebook", "--allow-root"]
ENTRYPOINT ["bash"]

