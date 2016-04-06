See the full documentation here:
<http://geoscienceaustralia.github.io/sifra/index.html>


What is it
===========
SIFRA is a **System for Infrastructure Facility Resilience Analysis**.
SIFRA comprises a method and software tools that provide a framework
for simulating the fragility of infrastructure facilities to natural
hazards, based on assessment of the fragilities and configuration of
components that comprises the facility. Currently the system is
designed to work with earthquake hazards only.

The following are some key features of this tool:

- Written in Python: It is written in Python, and there is no
  dependency on proprietary tools. It should run on OS X, Windows, and
  Linux platforms.
- Flexible Facility Model: ``facility`` data model is based on network
  theory, allowing the user to develop arbitrarily complex custom
  facility models for simulation.
- Extensible Component Library: User can define new ``component types``
  (the building blocks of a facility) and link it to existing or
  custom fragility algorithms.
- Component Criticality: Scenario Analysis tools allow users to
  identify the cost of restoration for chosen scenarios, expected
  restoration times, and which component upgrades can most benefit
  the system.
- Restoration Prognosis: User can experiment with different levels of
  hazards and post-disaster resource allocation to gauge restoration
  times for facility operations.


Setup Instructions
===================
It is good practice to set up a virtual environment for working with
developing code. This gives us the tools to manage the package
dependencies and requirements in a transparent manner, and impact of
dependency changes on software behaviour.


Preparing the Run Environment
------------------------------
We recommend using ``conda`` for managing virtual environments and
packages required for running ``sifra``.

For the sake of simplicity, we recommend using ``Anaconda``. It is a
free Python distribution, and comes with the ``conda`` tool which is
both a package manager and environment manager. Instructions for
installing ``Anaconda`` are here:
<http://docs.continuum.io/anaconda/install>

**Prerequisites:** You will need to install ``Graphviz`` for the 
drawing the system diagram through networkx and pygraphviz. 
Please visit: <http://www.graphviz.org/> 
and download the appropriate version for your operating system. 
Please follow the posted download instructions carefully. 
After installation you may need to update the PATH variable 
with the location of the Graphviz binaries.

For windows systems you will need to install 
Microsoft Visual C++ Compiler for Python 2.7:
<http://aka.ms/vcpython27>

Some packages we need are not hosted in the main ``conda`` package
repository. In such cases we will host them in our own user channel.
We suggest adding the following channels to the default::

    conda config --add channels https://conda.anaconda.org/anaconda
    conda config --add channels https://conda.anaconda.org/marufr

Run the following command to confirm the additional channels have
been added:

    conda config --get channels

**For OS X and Linux-64 systems:** It should be possible to set up a
full run environment solely through the *.yml environment specification
file. For OS X run the following commands:

    conda env create -f environment_osx.yml
    source activate sifra_env

For Linux-64 systems, the commands are identical, you will just need
to use the environment specification file for Linux.

**For Windows systems**, a similar process needs to be followed, with
some exceptions. First run:

    conda env create -f environment_win64.yml
    activate sifra_env

This will install most requirements except for ``igraph`` and
``pygraphviz``. Compiling these packages under windows can be very
challenging. The simplest and most reliable options is to download
the the appropriate wheels from Christoph Gohlke's unofficial page
of Windows binaries:
<http://www.lfd.uci.edu/~gohlke/pythonlibs/>

For Windows 64 bit systems, you will need to download the ``wheels`` for
[``python-igraph``](http://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph)
and [``pygraphviz``](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pygraphviz):
- ``python_igraph-0.7.1.post6-cp27-none-win_amd64.whl``
- ``pygraphviz-1.3.1-cp27-none-win_amd64.whl``

Install these downloaded ``wheels`` with pip:

    pip install <pkg_name>.whl


Running the Code
-----------------
*For the purposes of this discussion, we will assume that this
repository has been cloned in the user's home directory.*

First move into the root directory for the ``SIFRA`` code:

    cd sifra    # and not cd sifra/sifra

Run the `sifra` code as a module, with the requisite configuration
file:

    python -m sifra simulation_setup/config_file.conf

Depending on the scale of the model, and simulation paramters chosen,
it may take between a few minutes and a few days to complete a run.

To fit a system fragility for the facility to the simulated data
generated in the previous step, and a simple normal restoration
model, run the command:

    python sifra/fit_model.py simulation_setup/config_file.conf

To simulate the `component type` loss analysis, restoration prognosis,
and generate the component criticality plot, run the command:

    python sifra/scenario_loss_analysis.py simulation_setup/config_file.conf


Running the Tests
------------------
To run tests use either ``nose`` or ``unittest``.
Example (from the first level 'sifra' directory):

    cd sifra  # and not cd sifra/sifra
    python -m unittest discover tests

or, simply run:

    nosetest


:grey_exclamation: NOTE: Project needs a more comprehensive test suite.
