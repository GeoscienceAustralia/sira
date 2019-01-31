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

# install packages not avilble in /root/anaconda2/bin/conda 
RUN pip install alabaster==0.7.12 asn1crypto==0.24.0 awscli==1.16.96 Babel==2.6.0 backports-abc==0.5 backports.functools-lru-cache==1.5 botocore==1.12.86 brewer2mpl==1.4.1 certifi==2018.11.29 cffi==1.11.5 chardet==3.0.4 colorama==0.3.7 coloredlogs==9.0 configparser==3.5.0 cryptography==2.5 cycler==0.10.0 dask==1.1.0 decorator==4.3.2 docutils==0.14 enum34==1.1.6 functools32==3.2.3.post2 futures==3.2.0 humanfriendly==4.17 idna==2.8 imageio==2.4.1 imagesize==1.1.0 ipaddress==1.0.22 Jinja2==2.10 jmespath==0.9.3 latexcodec==1.0.6 lmfit==0.9.2 MarkupSafe==1.1.0 matplotlib==2.1.2 netCDF4==1.3.1 networkx==2.1 numpy==1.16.0 olefile==0.46 oset==0.1.3 packaging==19.0 pandas==0.22.0 parmap==1.5.1 patsy==0.5.1 Pillow==5.2.0 pyasn1==0.4.4 pybtex==0.21 pybtex-docutils==0.2.1 pycairo==1.18.0 pycparser==2.19 Pygments==2.2.0 pygraphviz==1.3.1 pyOpenSSL==18.0.0 pyparsing==2.3.1 PySocks==1.6.8 python-dateutil==2.7.5 python-igraph==0.7.1.post6 pytz==2018.9 PyWavelets==1.0.1 PyYAML==3.13 requests==2.21.0 rsa==3.4.2 s3transfer==0.1.13 scikit-image==0.13.1 scikit-learn==0.19.1 scipy==1.2.0 seaborn==0.8.1 singledispatch==3.4.0.3 six==1.11.0 snowballstemmer==1.2.1 Sphinx==1.7.0 sphinxcontrib-bibtex==0.4.2 sphinxcontrib-websupport==1.1.0 SQLAlchemy==1.2.1 statsmodels==0.8.0 subprocess32==3.5.3 toolz==0.9.0 tornado==5.1.1 typing==3.6.6 urllib3==1.24.1 xlrd==1.1.0  

ARG CACHE_DATE
RUN git clone --single-branch --branch  ft-update-cli https://github.com/GeoscienceAustralia/sifra.git
