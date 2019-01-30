# Delete all containers
docker rm $(docker ps -a -q)

# Delete all images
docker rmi $(docker images -q)

# Building image
docker build -t sifra . --build-arg CACHE_DATE="$(date)"
 
# Running a simulation
docker run -it --rm -v /home/ubuntu/scenario_dir:/scenario_dir sifra:latest $python sifra -d scenario_dir -sfl

# Running terminal in docker container 
docker run -it --rm sifra:latest python sifra
