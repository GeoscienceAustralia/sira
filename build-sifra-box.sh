#!/bin/bash

# This can be run as root, as would happen if it is used as user-data when
# spinning up a machine in AWS, or as another user. The next block sets prefixes
# for running commands for each case.
if [ "$(whoami)" == "root" ]; then
    SUDO=
    NOSUDO="sudo -i -u ubuntu"
else
    SUDO=sudo
    NOSUDO=
fi

$SUDO apt-get update
# should really say (except it is time consuming and can require user
# interaction as it stands)...
#$SUDO apt-get upgrade

# install prerequisites
$SUDO apt-get install -y \
    linux-image-extra-$(uname -r) \
    linux-image-extra-virtual \
    apt-transport-https \
    ca-certificates \
    curl \
    git \
    software-properties-common

# install docker
curl -fsSL https://apt.dockerproject.org/gpg | $SUDO apt-key add
$SUDO add-apt-repository \
       "deb https://apt.dockerproject.org/repo/ \
       ubuntu-$(lsb_release -cs) \
       main"
$SUDO apt-get update
$SUDO apt-get -y install docker-engine

# install docker-compose
$SUDO bash -c "curl -L https://github.com/docker/compose/releases/download/1.11.2/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose"
$SUDO chmod +x /usr/local/bin/docker-compose

# add ubuntu to the docker group to avoid constant sudoing
# note that if you log onto the box before this command is run, then it won't
# take effect. You would need to log off and then on again to have it do so.
$SUDO usermod -a -G docker ubuntu

# clone master from sifra.
# Of course, you would want to change the url of this once your ready to commit
# something.
$NOSUDO git clone https://github.com/GeoscienceAustralia/sifra.git /home/ubuntu/sifra

# get an awsome tmux configuration
curl https://raw.githubusercontent.com/Sleepingwell/tmux-conf/master/tmux.conf | $NOSUDO tee /home/ubuntu/.tmux.conf

