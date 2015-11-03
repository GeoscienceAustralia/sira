#### Virtual ennvironment setup
It is recommeneded that you create a virtualenv to run the sira code. These recommendations are for `ubuntu 14.04` and is expected to work for most newer versions of `ubuntu`. The virtualenv and the requirements can be installed using the following steps.

    sudo pip install virtualenv
    sudo apt-get -y build-dep matplotlib  # then endter you root password
    virtualenv -p python2.7 ~/sira 
    source ~/sira/bin/activate
    cd sira # and not cd sira/sira, this is where the requirements.txt exists
    pip install -r requirements.txt

Note in the above, the first commadn `sudo apt-get -y build-dep matplotlib` installs all the build dependencies 
for matplotlib.


#### How to run the SIRA code

Running the sira code is simple. First 
    
    cd sira # and not cd sira/sira

Then one can run the object oriented `sira.py` as     
    
    python -m sira simulation_setup/config_ps_X.conf

Or the procedural `sira_bk.py` as

    python sira/sira_bk.py simulation_setup/config_ps_X.conf

#### Run tests
To run tests one case use either `nose` or unittest like the following from the first level `sira` directory:
    
    cd sira  # and not cd sira/sira
    python -m unittest discover tests
    or
    nosetests



