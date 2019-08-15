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
