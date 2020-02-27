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
    && /opt/miniconda/bin/conda install conda=4.8.1=py37_0 \
    && /opt/miniconda/bin/conda clean -yq -a \
    && rm Miniconda3-4.7.12.1-Linux-x86_64.sh \
    && rm -rf \
        /tmp/* \
        /var/tmp/* \
        /usr/share/man \
        /usr/share/doc \
        /usr/share/doc-base

ENV PATH /opt/miniconda/bin:$PATH


RUN pip install jupyter ipdb -i  https://mirrors.aliyun.com/pypi/simple/

RUN  echo "10.16.94.96 github.com" >> /etc/hosts
RUN  echo "10.16.91.63 github.global.ssl.fastly.Net" >> /etc/hosts

RUN mkdir /app
COPY ./requirements.txt /app
RUN pip install -r /app/requirements.txt -i  https://mirrors.aliyun.com/pypi/simple/

USER root

WORKDIR /app

EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--allow-root"]