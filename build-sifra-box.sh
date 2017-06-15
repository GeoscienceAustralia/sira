#!/bin/bash

sudo apt-get update

# install prerequisites
sudo apt-get install -y \
    linux-image-extra-$(uname -r) \
    linux-image-extra-virtual \
    apt-transport-https \
    ca-certificates \
    curl \
    git \
    software-properties-common

# install docker
curl -fsSL https://apt.dockerproject.org/gpg | sudo apt-key add
sudo add-apt-repository \
       "deb https://apt.dockerproject.org/repo/ \
       ubuntu-$(lsb_release -cs) \
       main"
sudo apt-get update
sudo apt-get -y install docker-engine

# install docker-compose
sudo bash -c "curl -L https://github.com/docker/compose/releases/download/1.11.2/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose"
sudo chmod +x /usr/local/bin/docker-compose

# add ubuntu to the docker group to avoid constant sudoing
sudo usermod -a -G docker ubuntu

# clone master from sifra.
# Of course, you would want to change the url of this once your ready to commit
# something.
git clone https://github.com/GeoscienceAustralia/sifra.git ~/sifra

# get an awsome tmux configuration
curl https://raw.githubusercontent.com/Sleepingwell/tmux-conf/master/tmux.conf > ~/.tmux.conf

