# Delete all containers:
$ docker rm $(docker stop $(docker ps -aq))

# Delete all images:
$ docker rmi $(docker images --filter "dangling=true" -q)

# Building image:
$ docker build -t siraimg . --build-arg CACHE_DATE="$(date)"

# Run Option #1

The following command simulataneously does the following:
bind mounts a volume in docker, creates a container in interactive mode,
runs a simulation, then destroys the container after simulation ends.

$ docker run -it --rm -v /abs/local/path/<scenario_dir>:/<scenario_dir> \
    siraimg:latest \
    python sira -d <scenario_dir> -sfl --aws

# Run Option #2

$ docker create --name=sira_x -it siraimg:latest

Then:
$ docker start sira_x
$ docker attach sira_x

Or:
$ docker start -a -i sira_x

Run the sira code for the scenario in the specified directory:
$ python sira -d /path/to/scenario_dir -sfl

From outside of docker, on a terminal, use the following command to
copy the project folder from container to host:

$ docker cp $(docker ps -alq):/from/path/in/container /to/path/in/host/