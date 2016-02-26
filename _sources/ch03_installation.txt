
****************************
SIFRA Installation and Usage
****************************

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
through networkx and pygraphviz. Please visit: `<http://www.graphviz.org/>`_
and download the appropriate version for your operating system. Follow the 
posted download instructions carefully. After installation you may need to 
update the PATH variable with the location of the Graphviz binaries.

For windows systems you will need to install Microsoft Visual C++ Compiler 
for Python 2.7: <http://aka.ms/vcpython27>


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

**For OS X and Linux-64 systems:** It should be possible to set up a full run 
environment solely through the *.yml environment specification file. For OS X 
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
and most reliable options is to download the the appropriate wheels from 
`Christoph Gohlke's unofficial page of Windows binaries
<http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_.

For Windows 64 bit systems, you will need to download the ``wheels`` for
`python-igraph <http://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph>`_
and `pygraphviz <http://www.lfd.uci.edu/~gohlke/pythonlibs/#pygraphviz>`_:
- ``python_igraph-0.7.1.post6-cp27-none-win_amd64.whl``
- ``pygraphviz-1.3.1-cp27-none-win_amd64.whl``

Install these downloaded ``wheels`` with pip::

    pip install <pkg_name>.whl

.. _running-sifra:

Running the Code
================

To run the software, follow the following simple steps:

- Open a command terminal

- Change to the directory that has the ``sifra`` code. If the code is in the 
  directorty ``/Users/personx/sifra``, then run::

    cd ~/sifra/ 

- Run the primary fragility characterisarion module from the command line.
  We assume that same configuration file is stored in 
  ``/Users/personx/sifra/simulation_setup/``, and the configuration file is 
  names ``config_ps.conf``::

    python -m sifra simulation_setup/config_ps.conf


When the software is started for te first time, 
or if it cannot find a :ref:`scenario file <simulation-scenario-file>`, the 
user will be prompted to select a scenario file. If a scenario file has been 
loaded previously, SIFRA attempts to load that same scenario file then next 
time it is run.


Running Code Tests
==================

To run tests use either ``nose`` or ``unittest``.
Example (from the first level 'sifra' directory)::

    cd sifra  # and not cd sifra/sifra
    python -m unittest discover tests

or, simply run::

    nosetest


.. --- < begin : substitutions > ------------------------------------

.. |square-blk| unicode:: 0x25A0 .. Black square

.. |square-blk-sml| unicode:: 0x25AA .. Black square small

.. |square-wht| unicode:: 0x25A1 .. White square

.. |arrow-right| replace:: :math:`\Rightarrow`

.. --- < end : substitutions > --------------------------------------

