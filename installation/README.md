# Delete all containers
docker rm $(docker stop $(docker ps -aq))

# Delete all images
docker rmi $(docker images -qf "dangling=true")

# Building image
docker build -t sifra . --build-arg CACHE_DATE="$(date)"
 
# Running a simulation
docker run -it --rm -v /abs/path/on/local/machine/to/scenario_dir:/scenario_dir sifra:latest $python sifra -d scenario_dir -sfl --aws

# Run terminal inside docker container 
docker run -it --rm sifra:latest 