FROM continuumio/miniconda
RUN apt-get update && apt-get install -y \
    build-essential pkg-config \
    graphviz libgraphviz-dev \
    xml2 libxml2-dev
ADD environment_linux64.yml .
RUN conda env create -f environment_linux64.yml
RUN echo "source activate sifra_env" >> ~/.bashrc
VOLUME "/sifra"
ENV PYTHONPATH /sifra:$PYTHONPATH

