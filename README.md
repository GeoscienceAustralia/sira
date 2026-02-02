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
    + [Runtime Flags](#runtime-flags)
  - [Testing](#testing)
  - [Parallel Execution](#parallel-execution)
    + [HPC / MPI Environment Flags](#hpc-mpi-env-flags)

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

The dependency list is large. To assist in managing the list of required packages, they are sectioned into different files, based on the purpose they serve. For example, if documentation is not required to be generated, or experimental geospatial modules are not required, those files can be skipped.

The recommended process to set up the environment is to use `mamba` and `uv`.
This approach works equally well in Windows and Linux, and within Docker. The following script snippets assume the user is in the `sira` root directory. For setups using a combination of `mamba` and `pip` or `uv`, a consolidated pip requirements list is also provided.

Installation option #1 (necesary for Windows):

```
    mamba env create -f ./installation/sira_env.yml
    mamba activate sira_env
    pip install uv
    uv pip install -r ./installation/sira_req.txt
```

Installation option #2 (for Linux workstations, needs to be adapted for HPC env):

```
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

    + **model file**: it must have the term 'model' at the beginning of the file name
    + **config file**: it must have the term 'config' at the beginning of the file name

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

### [Runtime Flags](#runtime-flags)

SIRA recognises several environment flags to control behaviour related to detection and selection of the appropriate backend. Setting these flags is optional. These need to be set in the shell before running SIRA.

- `SIRA_ENABLE_GPU_DETECT`
    - Purpose: Enable optional GPU detection during environment setup.
    - Default: `0` (disabled). When set to `1`, SIRA will attempt to detect CUDA GPUs via PyTorch (if installed) or TensorFlow (if installed). Detection is informational only; SIRA does not currently perform GPU-accelerated computation.
    - Example (PowerShell):
        ```powershell
        $env:SIRA_ENABLE_GPU_DETECT = "1"
        python -c "from sira.parallel_config import ParallelConfig; ParallelConfig().print_config_summary()"
        ```

- `SIRA_FORCE_NO_MPI`
    - Purpose: Explicitly disable MPI detection and usage.
    - Default: `0` (not forced). When set to `1`, SIRA treats the environment as non-MPI even if MPI-related variables are present, and falls back to multiprocessing.
    - Example (PowerShell):
        ```powershell
        $env:SIRA_FORCE_NO_MPI = "1"
        ```

Notes:
- These flags only affect detection and backend selection. Core computations remain CPU-based unless an MPI backend is explicitly selected and available.
- Flags can be set per-session or integrated into CI/CD environment configuration.

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

### [HPC / MPI Environment Flags](#hpc-mpi-env-flags)

When running large-scale simulations on HPC (e.g. Gadi with PBS + OpenMPI), SIRA recognises additional environment flags. These flags are used in job scripts. These tune streaming, logging, batching and consolidation behaviour. All are optional; unset flags fall back to internal defaults.

- `SIRA_LOG_LEVEL`
    - Sets Python logger verbosity (e.g. `INFO`, `DEBUG`, `WARNING`).
    - Lower verbosity reduces I/O overhead in large parallel runs.
- `SIRA_QUIET_MODE`
    - `1` suppresses progress / non-essential console output; `0` shows normal progress.
- `SIRA_STREAM_DIR`
    - Directory for per-rank / per-process streamed intermediate artifacts (NPZ, manifests). Use fast local node storage (e.g. burst buffer) for performance; consolidate later.
- `SIRA_DEFER_CONSOLIDATION`
    - `1` skips in-process aggregation of streamed artifacts until a post-run consolidation step; improves runtime on large ranks.
- `SIRA_SAVE_COMPTYPE_RESPONSE`
    - `1` to persist component-type response arrays; `0` to skip. Saves aggregated loss metrics by type.
- `SIRA_SAVE_COMPONENT_RESPONSE`
    - `1` to persist per-component response arrays (larger footprint); `0` (default in scripts) to reduce storage.
- `SIRA_CHUNKS_PER_SLOT`
    - Controls how many streaming chunks each CPU slot (rank/worker) emits before rolling over to a new file. Lower values reduce memory overhead; higher values reduce file count.
- `SIRA_STREAM_COMPRESSION`
    - Compression codec for streamed artifacts (e.g. `snappy`, `zstd`). Choose fastest acceptable for your I/O profile.
- `SIRA_STREAM_ROW_GROUP`
    - Target row-group / block size (in rows) for streamed tabular data; balances read amplification vs memory. Large hazards benefit from larger values (e.g. `524288`).
- `SIRA_MIN_HAZARDS_FOR_PARALLEL`
    - Integer threshold; if total hazard events below this value, SIRA may avoid spawning full parallel workers to reduce overhead.
- `SIRA_HPC_MODE`
    - `1` enables HPC-oriented heuristics (larger batches, reduced chatter, defensive memory usage). When unset, defaults remain more general-purpose.
- `SIRA_MAX_BATCH_SIZE`
    - Caps batch size used in processing loops even if auto-tuning would choose larger; helps prevent memory spikes on dense component sets.
- `SIRA_CLEANUP_CHUNKS`
    - `1` removes staged per-node chunk directories after consolidation to reclaim scratch space; `0` keeps them for inspection.
- `PYTHONHASHSEED`
    - Standard Python reproducibility flag (e.g. set to `0`); ensures consistent hash-based ordering when determinism is required.

Example (PBS + OpenMPI snippet):
```bash
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
```

Recommendation:
- Start with only `SIRA_HPC_MODE=1` and `SIRA_STREAM_DIR` for large jobs; layer additional flags as bottlenecks are identified through profiling.
