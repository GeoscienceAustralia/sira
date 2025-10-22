Layered Python environment (HPC-friendly)
-----------------------------------------

We provide a minimal core requirements file suitable for headless HPC runs, and
an optional visualisation set for notebooks/plots. Critical pins live in a small
constraints file to keep installs reproducible but flexible.

Files:
- `installation/constraints.txt` – version pins for key libraries (NumPy, pandas, Dask, Numba, etc.).
- `installation/requirements-core.txt` – minimal runtime deps (includes python-igraph and scikit-learn).
- `installation/requirements-viz.txt` – optional plotting/jupyter extras (includes yellowbrick).

Create a venv on Gadi (HOME-based):

1) Create and activate the venv
    module load python3/3.11.7
    python3 -m venv $HOME/venv/sira-env
    source $HOME/venv/sira-env/bin/activate

2) Install constraints, core, and and other essential packages
    python -m pip install --upgrade pip
    python -m pip install -r installation/constraints.txt
    python -m pip install -r installation/requirements-core.txt
    python -m pip install -r installation/requirements-viz.txt
    python -m pip install -r installation/requirements-dev.txt
    python -m pip install -v --no-binary mpi4py mpi4py

3) Install mpi4py from source (for HPC environment)
    python3 -m pip install -v --no-binary :all: --user --cache-dir=$TMPDIR mpi4py

4) Optional extras
    python -m pip install -r installation/requirements-docs.txt
    python -m pip install -r installation/requirements-diagrams.txt
    python -m pip install -r installation/requirements-notebook.txt
    python -m pip install -r installation/requirements-geo.txt

5) Install SIRA
    python -m pip install -e .dea

Sanity check
    python - <<'EOF'
    import igraph, numpy, pandas, dask, distributed, sklearn
    print('Core imports OK')
    EOF

# Instructions for building a SIRA run environment in docker

## Building the docker image

Step 1: Delete all containers

    $ docker rm $(docker stop $(docker ps -aq))

Step 2: Delete all images

    $ docker rmi $(docker images --filter "dangling=true" -q)

Step 3: Build the docker image

    $ docker build -t siraimg . --build-arg CACHE_DATE="$(date)"

## Running SIRA in docker

### Run Option #1: Run a simulation and destroy the container when done

The following command simulataneously does the following:
bind mounts a volume in docker, creates a container in interactive mode, 
runs a simulation, then destroys the container after simulation ends.

    $ docker run -it --rm -v /abs/local/path/<scenario_dir>:/<scenario_dir> \
        siraimg:latest \
        python sira -d <scenario_dir> -sfl --aws

### Run Option #2: Build a container for reuse / experimentation

First, build a docker container from the prebuilt image.

    $ docker create --name=sira_x -it siraimg:latest

Then start and attach the container:

    $ docker start sira_x
    $ docker attach sira_x

It is possible to combine the above steps in one:

    $ docker start -a -i sira_x

Now, you can run the sira code for the scenario in the specified directory:

    $ python sira -d /path/to/scenario_dir -sfl

From outside of docker, on a terminal, use the following command to copy the project folder from container to host:

    $ docker cp $(docker ps -alq):/from/path/in/container /to/path/in/host/
