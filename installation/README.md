# Delete all containers:
$ docker rm $(docker stop $(docker ps -aq))

# Delete all images:
$ docker rmi $(docker images --filter "dangling=true" -q)

# Building image:
$ docker build -t sifraimg . --build-arg CACHE_DATE="$(date)"

# Run Option #1

The following command simulataneously does the following:
bind mounts a volume in docker, creates a container in interactive mode,
runs a simulation, then destroys the container after simulation ends.

$ docker run -it --rm -v /abs/local/path/<scenario_dir>:/<scenario_dir> \
    sifraimg:latest \
    python sifra -d <scenario_dir> -sfl --aws

# Run Option #2

$ docker create --name=sifra_x -it sifraimg:latest

Then:
$ docker start sifra_x
$ docker attach sifra_x

Or:
$ docker start -a -i sifra_x

Run the sifra code for the scenario in the specified directory:
$ python sifra -d /path/to/scenario_dir -sfl

From outside of docker, on a terminal, use the following comman to
copy the project folder from container to host:

$ docker cp $(docker ps -alq):/from/path/in/container /to/path/in/host/