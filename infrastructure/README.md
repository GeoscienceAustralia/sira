# AWS INFRASTRUCTURE

following are the points to remember:

policy for the s3 bucket containing the output bucket:

    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObject",
                "Resource": "arn:aws:s3:::wildrydes-syed-gardezi/*",
                "Condition": {
                    "IpAddress": {
                        "aws:SourceIp": "192.104.0.0/16"
                    }
                }
            }
        ]
    }
    

Installing Docker:

Doc: https://docs.docker.com/install/linux/docker-ce/ubuntu/

    # install docker 
    sudo apt-get remove docker docker-engine docker.io && 
    sudo apt-get update &&
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common && 
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && 
    sudo apt-key fingerprint 0EBFCD88 && 
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" && 
    sudo apt-get update && 
    sudo apt-get install docker-ce && 
    sudo usermod -a -G docker $USER 
    
    sudo docker build .
    sifra/infrastructure/dockerfile
    
    # test run machine
    docker run python:2.7 python --version
    
    # house keeping
    # delte all containers 
    docker container prune -f
    # delte specfic images 
    docker rmi  <image_name>