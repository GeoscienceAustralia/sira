
**********************
Installation and Usage
**********************

.. _system-requirements:

Requirements
============

**SIFRA** is built in Python 2.7. Python hardware requirements are fairly 
minimal. Most modern systems based on x86 architecture should be able to run 
this software.

SIFRA has been tested on the following operating systems:

- OS X 10.11
- Ubuntu 14.04 (64 bit)
- Windows 10

The code should work on most other versions of these operating systems, 
though the environment setup process will likely have some differences. 
Windows systems that are not based on the NT-platform are not supported. This 
restriction is due to the fact that from version 2.6 onwards Python has not 
supported non-NT Windows platforms. 

You will need to install ``Graphviz`` for the drawing the system diagram
through `networkx` and `pygraphviz` packages. Please visit:
`<http://www.graphviz.org/>`_ and download the appropriate version for
your operating system. Follow the posted download instructions carefully.
After installation you may need to update the PATH variable with the
location of the Graphviz binaries.

For windows systems you will need to install Microsoft Visual C++ Compiler 
for Python 2.7: `<http://aka.ms/vcpython27>`_


.. _setup-dev-environ:

Setting Up a Development Environment
====================================

We recommend using ``conda`` for managing virtual environments and
packages required for running ``sifra``.

For the sake of simplicity, we recommend using ``Anaconda``. It is a
free Python distribution, and comes with the ``conda`` tool which is
both a package manager and environment manager. Instructions for
installing ``Anaconda`` are
`here <http://docs.continuum.io/anaconda/install>`_.

Some packages we need are not hosted in the main ``conda`` package
repository. In such cases we will host them in our own user channel.
We suggest adding the following channels to the default::

    conda config --add channels https://conda.anaconda.org/anaconda
    conda config --add channels https://conda.anaconda.org/marufr

Run the following command to confirm the additional channels have
been added:

    conda config --get channels

**For OS X and Linux-64 systems**: It should be possible to set up a full run
environment solely through the \*.yml environment specification file. For OS X
run the following commands::

    conda env create -f environment_osx.yml
    source activate sifra_env

For Linux-64 systems, the commands are identical, you will just need to use 
the environment specification file for Linux.

**For Windows systems**, a similar process needs to be followed, with some 
exceptions. First run::

    conda env create -f environment_win64.yml
    activate sifra_env

This will install most requirements except for ``igraph`` and ``pygraphviz``. 
Compiling these packages under windows can be very challenging. The simplest 
and most reliable option is to download the the appropriate binary
distribution in the form of `wheels` from
`Christoph Gohlke's unofficial page of Windows binaries
<http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_.

Download the appropriate `wheels` (\*.whl files) of the following packages
for your Windows platform (32 or 64 bit):

- `python-igraph <http://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph>`_
- `pygraphviz <http://www.lfd.uci.edu/~gohlke/pythonlibs/#pygraphviz>`_.

Install the downloaded `wheels` (\*.whl files) with pip::

    pip install <pkg_name>.whl


.. _running-sifra:

Running the Core SIFRA Code
===========================

To run the software, follow the following simple steps:

1.  Open a command terminal

2.  Change to the directory that has the ``sifra`` code. If the code is
    in the directorty ``/Users/personx/sifra``, then run::

        cd ~/sifra/

3.  Run the primary fragility characterisarion module from the command line.
    We assume that same configuration file is stored in
    ``/Users/personx/sifra/simulation_setup/``, and the configuration file
    is names ``config_x.conf``::

        python -m sifra simulation_setup/config_x.conf

The post-processing tools are run as simple python scripts. It should be
noted, that the post-processing tools depend on the outputs produced by a
full simulation run that characterises the system fragility. Therefore,
thea full run of the SIFRA needs to be conducted on the system model of
interest prior to running the tools for model fitting and scenario and
restoration analysis tools. They are simply run as::

    cd ~/sifra/sifra/
    python fit_model.py ../simulation_setup/config_x.conf
    python scenario_loss_analysis.py ../simulation_setup/config_x.conf


Running Code Tests
==================

To run tests use either ``nose`` or ``unittest``.
Example (from the first level 'sifra' directory)::

    cd sifra  # and not cd sifra/sifra
    python -m unittest discover tests

or, simply run::

    nosetest

