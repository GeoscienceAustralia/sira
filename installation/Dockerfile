FROM ubuntu:16.04

# setup system variables
ENV TZ Australia/Sydeny
ENV DEBIAN_FRONTEND noninteractive
ENV AWS_DEFAULT_REGION ap-southeast-2
ENV AWS_DEFAULT_OUTPUT json
ENV ROOT_DIR /home/ubuntu/

ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV locale-gen en_US.UTF-8
ENV dpkg-reconfigure locales

# install system dependancies 
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils && apt-get install -y --no-install-recommends apt-utils && apt-get install -y build-essential && apt-get install -y pkg-config
RUN apt-get install -y graphviz libgraphviz-dev xml2 libxml2-dev git curl zip unzip jq gcc wget libfreetype6-dev libpng-dev libcairo2-dev libjpeg8-dev libpango1.0-dev libgif-dev g++
RUN apt update
RUN apt install -y libgl1-mesa-glx 

# install anaconda2 
RUN wget https://repo.continuum.io/archive/Anaconda2-2018.12-Linux-x86_64.sh 
RUN bash Anaconda2*.sh -b -p /root/anaconda2 
RUN rm Anaconda2*.sh 
RUN /root/anaconda2/bin/conda create -n pythonSifra python=2.7.11  -y

ENV PATH="/root/anaconda2/envs/pythonSifra/bin:$PATH"

# install packages not available in /root/anaconda2/bin/conda
RUN pip install -r requirements.txt

ARG CACHE_DATE
RUN git clone --single-branch --branch  ft-update-cli https://github.com/GeoscienceAustralia/sifra.git
