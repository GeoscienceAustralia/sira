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
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils
ADD packagelist_linux.txt .
RUN apt-get install -y  $(awk '{print $1}' packagelist_linux.txt)

# install anaconda2 
#RUN wget https://repo.continuum.io/archive/Anaconda2-2018.12-Linux-x86_64.sh
RUN wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
RUN bash Anaconda3*.sh -b -p /root/anaconda3
RUN rm Anaconda3*.sh
RUN /root/anaconda3/bin/conda create -n sifra_py3 python=3.7.2  -y

ENV PATH="/root/anaconda3/envs/sifra_py3/bin:$PATH"

# install packages not available in /root/anaconda2/bin/conda
ADD requirements.txt .
RUN pip install -r requirements.txt

ARG CACHE_DATE
RUN git clone --single-branch --branch  py3-sifra-migration https://github.com/GeoscienceAustralia/sifra.git
