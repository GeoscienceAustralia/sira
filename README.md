<a href="https://geoscienceaustralia.github.io/sira/"><img src="https://img.shields.io/badge/Docs-GitHub%20Pages-0c457d" alt="SIRA Documentation" /></a>
<a href="https://github.com/GeoscienceAustralia/sira/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT%2FApache--2.0-e8702a" alt="License: MIT / Apache 2.0" /></a>
<a href="https://github.com/GeoscienceAustralia/sira/actions/workflows/build-test-linux.yml"><img src="https://github.com/GeoscienceAustralia/sira/actions/workflows/build-test-linux.yml/badge.svg" alt="CI-LINUX"></a>
<a href="https://github.com/GeoscienceAustralia/sira/actions/workflows/build-test-win.yml"><img src="https://github.com/GeoscienceAustralia/sira/actions/workflows/build-test-win.yml/badge.svg" alt="CI-WIN" /></a>


# SIRA

- [SIRA](#sira)
  - [Overview](#overview)
  - [Setup Instructions](#setup-instructions)
    + [Build Environment](#build-environment)
    + [Required Directory Structure](#required-directory-structure)
  - [Running the Application](#running-the-application)
  - [Testing](#testing)
  - [Parallel Execution](#parallel-execution)

## [Overview](#oveview)

The detailed documentation see the related [gihub pages](https://geoscienceaustralia.github.io/sira/).

SIRA stands for **Systemic Infrastructure Resilience Analysis**.
It represents a methodology and supporting code for systematising vulnerability
analysis of lifeline infrastructure to natural hazards (i.e. response of
infrastructure assets to environmental excitation). SIRA is open source.

The impact assessment is based on the fragilities and configuration of
components that comprise the infrastructure system under study. The analytical
process is supplemented by an assessment of the system functionality through
the post-damage network flow analysis, and approximations for recovery
timeframes.

The current focus has been on studying responses of infrastructure facilities
(e.g. power generation plants, high voltage substations). Considerable work
has been done in the code backend to extend the same methodology to modelling
network vulnerability as well (e.g. electricity transmission networks).

SIRA models are effectively directed graphs. All infrastructure systems are
represented as networks. This allows an user to develop arbitrarily complex
models of a infrastructure facility or a network to be used in
impact simulation.


## [Setup Instructions](#setup-instructions)

It is good practice to set up a virtual environment for working with
developing code. This gives us the tools to manage the package
dependencies and requirements in a transparent manner, and impact of
dependency changes on software behaviour.

### [Build Environment](#build-environment)

The recommended process to set up the environment is to use `mamba` and `uv`.
This approach works equally well in Windows and Linux, and within Docker. Move
into the `sira/installation` directory, and use the provided conda environment file
(yaml) and `pip` requirements file (txt) to install the required packages:

```
    mamba env create -f sira_env.yml
    mamba activate sira_env
    pip install uv
    uv pip install -r sira_req.txt
```

### [Required Directory Structure](#required-directory-structure)

To set up a scenario or impact simulation project, SIRA expects the following
directory structure for the model to be run.

```
    model_dir
    │
    ├── input
    │   ├── config_assetx.json
    │   └── model_assetx.json
    └── output
        ├── ...
        └── ...
```

Notes on the required directory structure:

- **model directory**: it can be named anything.

- **input directory**: must reside within the 'model directory'. The input dir must have two files, and their naming must follow a the specified format:

    + **model file**: it must have the term 'model' at the beginning or
      end of the file name
    + **config file**: it must have the term 'config' at the beginning or
      end of the file name

- **output directory**: the outputs are saved in this dir.

    + If it does not exist, it will be created at the beginning of the simulation.
    + The default name is 'output' and default location is within the 'model directory'.
    + The user can define a custom name and relative location within the config file.

- **scenario file location**: If an event set is being used for the simulation, the location and name of the relevant file need to be specified in the parameters "HAZARD_INPUT_DIR" and "HAZARD_INPUT_FILE", respectively.


## [Running the Application](#running-the-application)

The application can be run in a number of modes. The relevant options are:

    -d <path_to_input_dir>, --input_directory <path_to_input_dir>
    -s, --simulation
    -f, --fit
    -l, --loss_analysis

The following code snippets assume that it is being run from the root
directory of the SIRA code, and the model of interest is in the location
`sira/scenario_dir/ci_model_x`.

The following code runs the simulation and the post processing simultanrously:

    python -m sira -d scenario_dir/ci_model_x -sfl

To run only the Monte Carlo simulation without post-processing:

    python -m sira -d scenario_dir/ci_model_x -s

To run both the model fitting and the loss analysis code:

    python -m sira -d scenario_dir/ci_model_x -fl

Note that the model fitting and loss analysis steps require that the
initial simulation be run first so that it has the initial output data
to perform the analysis on.

## [Testing](#testing)

To run the tests, user needs to be in the root directory of the code,
e.g. `~/code/sira`. Then simply run:

    pytest

If you want to explicitly ask `pytest` to run coverage reports, then run:

    pytest --cov-report term --cov=sira tests/

If you are using docker as described above, you can do this from within the
sira container.

## [Parallel Execution](#parallel-execution)

SIRA supports multiprocessing locally and MPI on HPC systems. The MPI backend is preferred when available; otherwise, SIRA uses efficient multiprocessing with sensible defaults.

Note on selecting the parallel backend:

- Default (auto-detect): if launched under an MPI environment (e.g., mpirun/mpiexec/srun or SLURM/PBS variables present), SIRA uses MPI; otherwise it uses multiprocessing.

    python -m sira -d scenario_dir/ci_model_x -s --parallel-backend auto

- Force MPI: requires launching with an MPI launcher (and mpi4py installed). Example:

    mpirun -n 8 python -m sira -d scenario_dir/ci_model_x -s --parallel-backend mpi

- Force multiprocessing: runs locally without MPI. You can cap workers with --max-workers:

    python -m sira -d scenario_dir/ci_model_x -s --parallel-backend multiprocessing --max-workers 8

- Disable parallel entirely (useful for debugging):

    python -m sira -d scenario_dir/ci_model_x -s --disable-parallel

Optional tuning:

- `--scenario-size auto|small|medium|large|xlarge` lets SIRA tune defaults when no config file is provided (auto is recommended).

- You can also provide a JSON config via `--parallel-config` to pin backend and worker counts.
