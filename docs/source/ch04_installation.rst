.. _installation-usage:

**********************
Installation and Usage
**********************

**SIRA** is built on Python 3.x. Python hardware requirements are fairly
minimal. Most modern systems based on x86 architecture should be able to
run this software.

The directory structure of the code is as follows::

    .
    ├── docs                        <-- Sphinx documentation files
    │   └── source
    ├── hazard                      <-- Hazard scenario files (for networks)
    ├── installation                <-- Installation scripts for dev envs
    ├── scripts
    ├── sira                        <-- The core code reside here
    │   └── modelling
    ├── tests                       <-- Test scripts + data for sanity checks
    │   ├── historical_data
    │   ├── models
    │   └── simulation_setup
    │
    ├── LICENSE                      <-- License file
    ├── README.md                    <-- Summary documentation
    ├── __main__.py                  <-- Entry point for running the code
    ├── LICENSE
    └── README.adoc                  <-- Basic documentation and installation notes


Requirements
============

SIRA has been tested on the following operating systems:

    - OS X 10.11+
    - Ubuntu 14.04 (64 bit)
    - Windows 10

The code should work on other recent versions of these operating systems,
though the environment setup process may have some differences.
Windows systems that are not based on the NT-platform are not supported.
This restriction is due to the fact that from version 2.6 onwards Python
has not supported non-NT Windows platforms. 

You will need to install ``Graphviz``, which is used by
``networkx`` and ``pygraphviz`` packages to draw the system diagram.
Please visit: `<https://www.graphviz.org/>`_ and download the appropriate
version for your operating system. Follow the posted download instructions
carefully. After installation you may need to update the PATH variable
with the location of the Graphviz binaries.

For windows systems you will need to have a C++ Compiler. The recommended
option for Python 3.5+ is the free full-featured community edition of 
`Visual Studio 2015
<https://www.visualstudio.com/en-us/products/visual-studio-community-vs.aspx>`_.


.. _setup-dev-environ:

AWS and Docker
==============

The development of the Web interface to provide the ability to create
component types and models led to the usage of Docker and AWS. Docker
creates containers that provide independence of platforms when developing
applications and services. So Docker removes the requirement for Conda
to organise the Python libraries. The downside of using docker in our
environment (GA Sysmonston) is that network security makes running Docker
difficult. So development environments are easier to create and use on AWS.
Use of AWS for Web applications is the current direction for GA.

Installation of SIRA on AWS with Docker
+++++++++++++++++++++++++++++++++++++++

This requires building an AWS instance and then building the environments
using Docker.

**Create AWS Instance using Packer:** |br|
We're we come from, if we don't have a laptop handy, we like to use AWS for
provisioning dev machines. A basic dev box can be setup using
`Packer <https://www.packer.io/intro/>`_, by running::

    $ packer build build.json

in the current directory.

**Create AWS Instance using bash script:** |br|
The top level directory of SIRA has the script ``create-aws-instance.sh``
The script requires the `aws command line interface <https://aws.amazon.com/cli/>`_
to be installed on the machine. It also requires access to AWS account
credentials.

The script is run as::

    $ create-aws-instance.sh

Both of these commands will use the build_sira_box.sh to install Linux updates
and the docker components that will be required.

It then clones the SIRA github repository, from the master branch. Docker is
then used to build the SIRA environment.

Using Docker
++++++++++++

If you have Docker installed, you can build a container for working with
sira by running the command::

    $ docker build -t sira .

The primary advantage of working with docker is that you do not have to worry
about setting up the python environment, which is done when building the
container and isolated from your own environment.

To run an interactive container you can use::

    $ docker run -it -v "$(pwd):/sira" --name sira s

This will give you a terminal inside the container in which you can execute
commands. Inside the container you can find the current directory mapped at
`/sira`. You can modify files either within the container or the host and the
changes will be available in both.

Alternatively, you might want a container running in the background which you
can execute commands at using
`docker exec <https://docs.docker.com/engine/reference/commandline/exec/>`_. In
this case you would start the container with::

    $ docker run -id -v "$(pwd):/sira" --name sira sira

One could then, for example, run the unit tests for the modelling package with::

    $ cd sira/tests
    $ python -m unittest discover .

In any case, once you are done you should destroy the container with::

    $ docker kill sira
    $ docker rm sira


or, if your too lazy to type two lines, use this command::

    $ docker rm -f sira

Several other containers are provided to help with development. These are
defined in the other `Dockerfiles` in the present directory, and are:

- ``Dockerfile-api``: |br|
  Provides a web API which is used for parameterising
  model components (at this stage just response functions) and serialising them.
  This is presently (at Feb 2018) a prototype and provides only a small subset
  of what we hope for.

- ``Dockerfile-gui-dev``: |br|
  Provides an `Angular2 <https://angular.io/>`_ application for
  defining model components built on top of the API mentioned above. The application
  is hosted using Angular's development server and can be accessed on *localhost:4200*.

- ``Dockerfile-gui-prod``: |br|
  For deploying the web application in production. This
  does a production build of the Angular project and hosts it using
  `busybox <https://www.busybox.net/>`_. The app is still exposed on port 4200,
  so to host it at port 80 one would start it with::

    $ docker build -t sira-gui -f Dockerfile-gui-prod .

and start it with (for example)::

    $ docker run -d -p 80:4200 --restart always sira-gui-prod

Docker Compose
++++++++++++++

By far the easiest way to run the system for development is with
`docker-compose <https://docs.docker.com/compose/>`_, which can be done with::

    $ docker-compose up

Assuming that you start the system this way in the current folder, you can:

- attach to the sifa image to run models and tests with: |br|
  ``$ docker attach sira_sira_1``


- access the GUI for defining fragility functions at: |br|
  ``http://localhost:4200``, and


- access the web API at: |br|
  ``http://localhost:5000``.


This method will allow both the API and GUI to stay in sync with your code.

You can tear the system down (destroying the containers) with::

    $ docker-compose down

Setting Up a Development Environment with Anaconda
==================================================

We recommend using ``conda`` for managing virtual environments and
packages required for running ``sira``.

For the sake of simplicity, we recommend using `Anaconda`. It is a
free Python distribution, and comes with the ``conda`` tool which is
both a package manager and environment manager. Instructions for
installing `Anaconda` are
`here <http://docs.continuum.io/anaconda/install>`_.

Some packages we need are not hosted in the main ``conda`` package
repository. In such cases we will host them in our own user channel.
We suggest adding the following channels to the default::

    $ conda config --add channels https://conda.anaconda.org/anaconda
    $ conda config --add channels https://conda.anaconda.org/marufr

Run the following command to confirm the additional channels have
been added::

    $ conda config --get channels

**For OS X and Linux-64 systems**: It should be possible to set up a full run
environment solely through the \*.yml environment specification file. For OS X
run the following commands::

    $ conda env create -f environment_osx.yml
    $ source activate sira_env

For Linux-64 systems, the commands are identical, you will just need to use 
the environment specification file for Linux.

**For Windows systems**, a similar process needs to be followed, with some 
exceptions. First run::

    $ conda env create -f environment_win64.yml
    $ activate sira_env

This will install *most* requirements except for ``igraph`` and ``pygraphviz``.
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


.. _running-sira:


Running a Simulation with SIRA
==============================

The code needs a setup file for configuring the model and simulation scenario.
It needs to be in JSON format. It *can* support any of three formats:
`ini`, `conf`, or `json`. But details will not be discussed here. It also
requires a model defintion file as discussed in :ref:`simulation-input-setup`.

For the purposes of discussion, it is assumed that the name of the project
simulation directory is 'PROJECT_HAN', located in the root directory. 
The system name assumed is 'SYSTEM_GISKARD'.

The software can be run from the command line using these simple steps:

1.  Open a command terminal

2.  Change to the directory that has the ``sira`` code. Assuming the code is
    in the directorty ``/Users/user_x/sira``, run::

        $ cd ~/sira/

3.  Run the primary system fragility characterisation module from the
    command line using the following command::

        $ python sira -d ./PROJECT_HAN/SYSTEM_GISKARD/ -s

The code must be provided the full or relative path to the project
directorty that holds the input dir with the required config and model files.

The post-processing tools are run as simple python scripts. It should be
noted, that the post-processing tools depend on the outputs produced by a
full simulation run that characterises the system fragility. Therefore,
the full run of the SIRA needs to be conducted on the system model of
interest prior to running the tools for the loss scenario and
restoration analysis tools.

To run the post-simulation analysis on the generated output data, we need to
supply the flaf `-f` for model fitting, and the flag `-l` for loss analysis.
The flags can be combined.

To run the characterisation simulation, followed by model fitting, and
loss and recovery analysis, the command is::

        $ python sira -d ./PROJECT_HAN/SYSTEM_DANEEL/ -sfl


Running Code Tests
==================

To run tests use ``unittest``. The tests need to be run from the root of
the `sira` code directory::

    $ cd sira   # and not $ cd sira/sira
    $ python -m unittest discover tests

If you are using docker as described above, you can do this within the sira
container.
