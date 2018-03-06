#!/bin/bash

# This can be run as root, as would happen if it is used as user-data when
# spinning up a machine in AWS, or as another user. The next block sets prefixes
# for running commands for each case.
if [ $(id -u) -eq 0 ]; then
    SUDO=
    NOSUDO="sudo -i -u ubuntu"
else
    SUDO=sudo
    NOSUDO=
fi

# Older versions of Docker were called docker or docker-engine. If these are installed, uninstall them:
$SUDO apt-get remove docker docker-engine docker.io

# Update the apt package index:
$SUDO apt-get update
# should really say (except it is time consuming and can require user
# interaction as it stands)...
#$SUDO apt-get upgrade

# install prerequisites
$SUDO apt-get install -y \
    # TODO check if needed
    linux-image-extra-$(uname -r) \
    # TODO check if needed
    linux-image-extra-virtual \
    apt-transport-https \
    ca-certificates \
    curl \
    git \
    software-properties-common

# install docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$SUDO add-apt-repository \
       "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
       $(lsb_release -cs) \
       stable"
       
# Update the apt package index.
$SUDO apt-get update

# Install the latest version of Docker CE
$SUDO apt-get install docker-ce
# TODO for production define the version number
#$SUDO apt-get install docker-ce=17.12.0-ce

# Download the latest version of Docker Compose
$SUDO bash -c "curl -L https://github.com/docker/compose/releases/download/1.19.0/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose"

# Apply executable permissions to the binary
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
