#### How to run the SIRA code

Running the sira code is simple. First 
    
    cd sira # and not cd sira/sira

Then one can run the object oriented `sira.py` as     
    
    python -m sira simulation_setup/config_ps_X.conf

Or the procedural `sira_bk.py` as

    python sira/sira_bk.py simulation_setup/config_ps_X.conf

#### Virtual ennvironment setup
The requirements can be installed using the following steps.

    sudo apt-get -y build-dep matplotlib  # then endter you root password
    virtualenv -p python2.7 /home/sudipta/Dropbox/GA/CODE/igraphtest
    source /home/sudipta/Dropbox/GA/CODE/igraphtest/bin/activate

Note in the above, the first commadn `sudo apt-get -y build-dep matplotlib` installs all the build dependencies 
for matplotlib.

#### Run tests
To run tests one case use either `nose` or unittest like the following from the first level `sira` directory:
    
    cd sira  # and not cd sira/sira
    python -m unittest discover tests
    or
    nosetests



