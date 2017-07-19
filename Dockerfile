FROM ubuntu
RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y pkg-config
RUN apt-get install -y graphviz libgraphviz-dev
RUN apt-get install -y xml2 libxml2-dev
RUN apt-get install -y python-pip python-dev
RUN apt-get install -y python-tk
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas
RUN pip install networkx
RUN pip install seaborn
RUN pip install scipy
RUN pip install matplotlib
RUN pip install lmfit
RUN pip install brewer2mpl
RUN pip install colorama
RUN pip install parmap
RUN pip install python-igraph==0.7.1.post6
RUN pip install pygraphviz
RUN pip install xlrd
RUN pip install jgraph
RUN pip install sqlalchemy
RUN pip install Sphinx
RUN pip install sphinx-rtd-theme
RUN pip install guzzle-sphinx-theme
RUN pip install sphinxcontrib-actdiag
RUN pip install sphinxcontrib-bibtex
RUN pip install sphinxcontrib-blockdiag
RUN pip install sphinxcontrib-exceltable
RUN pip install sphinxcontrib-nwdiag
RUN pip install sphinxcontrib-seqdiag
RUN pip install sphinxcontrib-websupport
RUN pip install Pygments
RUN pip install pybtex
RUN pip install scikit-learn
RUN pip install six
VOLUME "/sifra"
WORKDIR "/sifra/sifra"
ENV PYTHONPATH /sifra:$PYTHONPATH

