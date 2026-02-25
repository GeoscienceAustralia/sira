.. _installation-usage:

**********************
Installation and Usage
**********************

**SIRA** is built on Python 3.x. Python hardware requirements are fairly
minimal. Most modern systems based on x86 architecture should be able to
run this software.

The directory structure of the code is as follows:

::

    .
    ├── docs                        <-- Sphinx documentation files
    │   └── source
    ├── hazard                      <-- Hazard scenario files (for networks)
    ├── installation                <-- Installation scripts for dev envs
    ├── scripts
    ├── sira                        <-- The core code reside here
    │   ├── __main__.py             <-- Entry point for running the code
    │   ├── modelling
    │   ├── scripts
    │   └── tools
    ├── tests                       <-- Test scripts + data for sanity checks
    │   ├── historical_data
    │   ├── models
    │   └── simulation_setup
    │
    ├── LICENSE                      <-- License file
    ├── pyproject.toml               <-- Project configuration file
    └── README.md                    <-- Summary documentation and usage notes


Requirements
============

SIRA has been tested on the following operating systems:

    - OS X 10.11+
    - Ubuntu 14.04 (64 bit)
    - Windows 10 and 11 (64 bit)

The code should work on more recent versions of these operating systems,
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


.. _recommended-installation:

Recommended Process for Building a SIRA Environment
===================================================

The recommended process to set up the environment is to use ``mamba`` 
and ``uv``. This approach works equally well in Windows and Linux, and 
within Docker. The following script snippets assume the user is in the 
``sira`` root directory.

The dependency list is large. To assist in managing the list of required 
packages, they are sectioned into different files, based on the purpose they 
serve. For example, if documentation is not required to be generated, or 
experimental geospatial modules are not required, those files can be skipped. 
The dependency files are provided in the ``installation`` directory. For 
setups using a combination of ``mamba`` and ``pip`` or ``uv``, a consolidated 
pip requirements list is also provided.

Instructions for installing `mamba` can be found
`here <http://docs.continuum.io/anaconda/install>`_.

Installation option #1 (necesary for Windows):

.. code-block:: bash

    mamba env create -f ./installation/sira_env.yml
    mamba activate sira_env
    pip install uv
    uv pip install -r ./installation/sira_req.txt


Installation option #2 (for Linux workstations, needs to be adapted for HPC env):

.. code-block:: bash

    sudo apt-get update
    grep -vE '^\s*#|^\s*$' ./installation/packagelist_linux.txt | \
    xargs -r sudo apt-get install -y --no-install-recommends
    sudo apt-get clean

    sudo rm -rf /var/lib/apt/lists/*
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt install python3.11.7

    python -m venv .venv
    python -m pip install --upgrade pip

    pip install -r ./installation/constraints.txt
    pip install -r ./installation/requirements-core.txt
    pip install -r ./installation/requirements-dev.txt
    pip install -r ./installation/requirements-viz.txt
    pip install -r ./installation/requirements-docs.txt
    pip install -r ./installation/requirements-diagrams.txt


.. _dev-env-setup:

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
---------------------------------------

This requires building an AWS instance and then building the environments
using Docker.

**Create AWS Instance using Packer:** |br|
We're we come from, if we don't have a laptop handy, we like to use AWS for
provisioning dev machines. A basic dev box can be setup using
`Packer <https://www.packer.io/intro/>`_, by running::

    packer build build.json

in the current directory.

**Create AWS Instance using bash script:** |br|
The top level directory of SIRA has the script ``create-aws-instance.sh``
The script requires the `aws command line interface <https://aws.amazon.com/cli/>`_
to be installed on the machine. It also requires access to AWS account
credentials.

The script is run as::

    create-aws-instance.sh

Both of these commands will use the build_sira_box.sh to install Linux updates
and the docker components that will be required.

It then clones the SIRA github repository, from the master branch. Docker is
then used to build the SIRA environment.

Using Docker
------------

If you have Docker installed, you can build a container for working with
sira by running the command::

    docker build -t sira .

The primary advantage of working with docker is that you do not have to worry
about setting up the python environment, which is done when building the
container and isolated from your own environment.

To run an interactive container you can use::

    docker run -it -v "$(pwd):/sira" --name sira s

This will give you a terminal inside the container in which you can execute
commands. Inside the container you can find the current directory mapped at
`/sira`. You can modify files either within the container or the host and the
changes will be available in both.

Alternatively, you might want a container running in the background which you
can execute commands at using
`docker exec <https://docs.docker.com/engine/reference/commandline/exec/>`_. In
this case you would start the container with::

    docker run -id -v "$(pwd):/sira" --name sira sira

One could then, for example, run the unit tests for the modelling package with::

    cd sira/tests
    python -m unittest discover .

In any case, once you are done you should destroy the container with::

    docker kill sira
    docker rm sira

or, simply run::

    docker rm -f sira

Several other containers are provided to help with development. These are
defined in the other `Dockerfiles` in the present directory, and are:

`Dockerfile-api`
  Provides a web API which is used for parameterising model components
  (at this stage just response functions) and serialising them.
  This is presently (at Feb 2018) a prototype and provides only a
  small subset of what we hope for.

`Dockerfile-gui-dev`
  Provides an `Angular2 <https://angular.io/>`_ application for
  defining model components built on top of the API mentioned above.
  The application is hosted using Angular's development server and can
  be accessed on *localhost:4200*.

`Dockerfile-gui-prod`
  For deploying the web application in production. This
  does a production build of the Angular project and hosts it using
  `busybox <https://www.busybox.net/>`_. The app is still exposed on
  port 4200, so to host it at port 80 one would start it with::

    docker build -t sira-gui -f Dockerfile-gui-prod .

  and start it with (for example)::

    docker run -d -p 80:4200 --restart always sira-gui-prod

Docker Compose
--------------

By far the easiest way to run the system for development is with
`docker-compose <https://docs.docker.com/compose/>`_, which can be done with::

    docker-compose up

Assuming that you start the system this way in the current folder, you can:

- attach to the `sira` image to run models and tests with: 
  ``docker attach sira``

- access the GUI for defining fragility functions at: 
  ``http://localhost:4200``

- access the web API at: 
  ``http://localhost:5000``


This method will allow both the API and GUI to stay in sync with your code.

You can tear the system down (destroying the containers) with::

    docker-compose down


.. _running-sira:

Running a Simulation with SIRA
==============================

The code needs a simulation setup file and an infrastructure model file
to run a simulation, as discussed in :ref:`simulation-input-setup`.

For the purposes of discussion, it is assumed that the name of the project
simulation directory is 'PROJECTX', located in the root directory. 
The system name assumed is 'SYSTEM_D'.

The software can be run from the command line using these simple steps:

1.  Open a command terminal

2.  Change to the directory that has the ``sira`` code. Assuming the code is
    in the directorty ``/Users/user_x/sira``, run::

        cd ~/sira/

3.  Run the primary system fragility characterisation module from the
    command line using the following command::

        python sira -d ./PROJECTX/SYSTEM_D/ -s

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

        python sira -d ./PROJECTX/SYSTEM_D/ -sfl

.. _running-tests:

Running Code Tests
------------------

After installation, it would be prudent to run the suite of tests to ensure everything is working correctly. To run the tests, users need to be in the root directory of the code, e.g. `~/code/sira`. Then simply run:

.. code-block:: bash

    pytest

If you want to explicitly ask `pytest` to run coverage reports, then run:

.. code-block:: bash

    pytest --cov-report term --cov=sira tests/

If you are using docker as described above, you can do this from within the
sira container.


.. _runtime-flags:

Runtime Flags
-------------

SIRA recognises several environment flags to control behaviour related to detection and selection of the appropriate backend. Setting these flags is optional. These need to be set in the shell before running SIRA.

`SIRA_ENABLE_GPU_DETECT`
    - Purpose: Enable optional GPU detection during environment setup.
    - Default: `0` (disabled). When set to `1`, SIRA will attempt to detect CUDA GPUs via PyTorch (if installed) or TensorFlow (if installed). Detection is informational only; SIRA does not currently perform GPU-accelerated computation.
    - Example (PowerShell):

    .. code-block:: powershell

        $env:SIRA_ENABLE_GPU_DETECT = "1"
        python -c "from sira.parallel_config import ParallelConfig; ParallelConfig().print_config_summary()"

`SIRA_FORCE_NO_MPI`
    - Purpose: Explicitly disable MPI detection and usage.
    - Default: `0` (not forced). When set to `1`, SIRA treats the environment as non-MPI even if MPI-related variables are present, and falls back to multiprocessing.
    - Example (PowerShell):

    .. code-block:: powershell

        $env:SIRA_FORCE_NO_MPI = "1"


.. note::

    - These flags only affect detection and backend selection.
      Core computations remain CPU-based unless an MPI backend is 
      explicitly selected and available.

    - Flags can be set per-session or integrated into CI/CD environment configuration.


.. _parallel-execution:

Parallel Execution
==================

SIRA supports multiprocessing locally and MPI on HPC systems. The MPI backend is preferred when available; otherwise, SIRA uses efficient multiprocessing with sensible defaults.

Notes on selecting the parallel backend:

- Default (auto-detect): if launched under an MPI environment (e.g., mpirun/mpiexec/srun or SLURM/PBS variables present), SIRA uses MPI; otherwise it uses multiprocessing::

    python -m sira -d scenario_dir/ci_model_x -s --parallel-backend auto

- Force MPI: requires launching with an MPI launcher (and mpi4py installed). Example::

    mpirun -n 8 python -m sira -d scenario_dir/ci_model_x -s --parallel-backend mpi


- Force multiprocessing: runs locally without MPI. You can cap workers with --max-workers::

    python -m sira -d scenario_dir/ci_model_x -s --parallel-backend multiprocessing --max-workers 8

- Disable parallel entirely (useful for debugging)::

    python -m sira -d scenario_dir/ci_model_x -s --disable-parallel


Optional tuning:

- Tune SIRA defaults using `--scenario-size auto|small|medium|large|xlarge` when no config file is provided (auto is recommended).

- There is an option to explicitly provide a JSON config via `--parallel-config` to pin backend and worker counts.

.. _hpc-mpi-flags:

HPC / MPI Environment Flags
---------------------------

When running large-scale simulations on HPC (e.g. Gadi with PBS + OpenMPI), SIRA recognises additional environment flags. These flags are used in job scripts. These tune streaming, logging, batching and consolidation behaviour. All are optional; unset flags fall back to internal defaults.

`SIRA_LOG_LEVEL`
    - Sets Python logger verbosity (e.g. `INFO`, `DEBUG`, `WARNING`).
    - Lower verbosity reduces I/O overhead in large parallel runs.

`SIRA_QUIET_MODE`
    - `1` suppresses progress / non-essential console output.
    - `0` shows normal progress.

`SIRA_STREAM_DIR`
    - Directory for per-rank / per-process streamed intermediate artifacts (NPZ, manifests).
    - Use fast local node storage (e.g. burst buffer) for performance; consolidate later.

`SIRA_DEFER_CONSOLIDATION`
    - `1` skips in-process aggregation of streamed artifacts until a post-run consolidation step.
    - Improves runtime on large ranks.

`SIRA_SAVE_COMPTYPE_RESPONSE`
    - `1` to persist component-type response arrays; `0` to skip.
    - Saves aggregated loss metrics by type.

`SIRA_SAVE_COMPONENT_RESPONSE`
    - `1` to persist per-component response arrays (larger footprint).
    - `0` to reduce storage (default in scripts).

`SIRA_CHUNKS_PER_SLOT`
    - Controls how many streaming chunks each CPU slot (rank/worker) emits before rolling over to a new file.
    - Lower values reduce memory overhead; higher values reduce file count.

`SIRA_STREAM_COMPRESSION`
    - Compression codec for streamed artifacts (e.g. `snappy`, `zstd`).
    - Choose fastest acceptable for your I/O profile.

`SIRA_STREAM_ROW_GROUP`
    - Target row-group / block size (in rows) for streamed tabular data; balances read amplification vs memory.
    - Large hazards benefit from larger values (e.g. `524288`).

`SIRA_MIN_HAZARDS_FOR_PARALLEL`
    - Integer threshold for number of hazard events to enalble parallel processing.
    - If total hazard events below this value, SIRA may avoid spawning full parallel workers to reduce overhead.

`SIRA_HPC_MODE`
    - `1` enables HPC-oriented heuristics (larger batches, reduced chatter, defensive memory usage).
    - When unset, defaults remain more general-purpose.

`SIRA_MAX_BATCH_SIZE`
    - Caps batch size used in processing loops even if auto-tuning would choose larger
    - Helps prevent memory spikes on dense component sets.

`SIRA_CLEANUP_CHUNKS`
    - `1` removes staged per-node chunk directories after consolidation to reclaim scratch space.
    - `0` keeps them for inspection.

`PYTHONHASHSEED`
    - Standard Python reproducibility flag (e.g. set to `0`).
    - Ensures consistent hash-based ordering when determinism is required.


Example (PBS + OpenMPI snippet):

.. code-block:: bash

    export SIRA_LOG_LEVEL=INFO
    export SIRA_QUIET_MODE=1
    export SIRA_STREAM_DIR="/iointensive/sira_${PBS_JOBID}"
    export SIRA_DEFER_CONSOLIDATION=1
    export SIRA_SAVE_COMPTYPE_RESPONSE=1
    export SIRA_SAVE_COMPONENT_RESPONSE=0
    export SIRA_CHUNKS_PER_SLOT=1
    export SIRA_STREAM_COMPRESSION=snappy
    export SIRA_STREAM_ROW_GROUP=524288
    export SIRA_MIN_HAZARDS_FOR_PARALLEL=100000
    export SIRA_HPC_MODE=1
    export SIRA_MAX_BATCH_SIZE=1000
    export PYTHONHASHSEED=0


The recommendation is to start only with `SIRA_HPC_MODE=1` and `SIRA_STREAM_DIR` for large jobs, identify bottlenecks through profiling or HPC logs, and then layer additional flags as needed.
