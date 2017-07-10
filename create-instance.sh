#!/bin/bash

# $1: The name of the key-pair to use.
# $2: The "Name" to give the instance.

# AMI is ubuntu in Sydney (at time of writing).
# Image size of medium seems to be the least we can get away with.

aws ec2 run-instances \
    --image-id ami-96666ff5 \
    --count 1 \
    --instance-type t2.medium \
    --user-data file://build-sifra-box.sh \
    --key-name $1 \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$2}]"

