#!/bin/bash

# $1: The name of the key-pair to use.

# AMI is ubuntu in Sydney (at time of writing).
# Image size of medium seems to be the least we can get away with.

# get the name of the latest ubuntu
ami_id=$(aws ec2 describe-images \
    --owners 099720109477 \
    --query 'reverse(Images.sort_by([], &CreationDate))[0].[ImageId][0]' \
    --output text \
    --filters \
        'Name=root-device-type,Values=ebs' \
        'Name=architecture,Values=x86_64' \
        'Name=virtualization-type,Values=hvm' \
        'Name=name,Values=*xenial-16.04-amd64-server*')



aws ec2 run-instances \
    --image-id $ami_id \
    --count 1 \
    --iam-instance-profile Name=SifraInstance \
    --instance-type t2.2xlarge \
    --user-data file://build-sifra-box.sh \
    --key-name $1 \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=sifra-$1}]"
