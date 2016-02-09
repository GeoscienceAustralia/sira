#### Virtual ennvironment setup
It is recommended that you create a `virtualenv` to run the `sifra` code. These instructions are for `ubuntu 14.04` and is expected to work for most newer versions of `ubuntu`. The virtualenv and the requirements can be installed using the following steps.

    sudo pip install virtualenv
    sudo apt-get -y build-dep matplotlib  # then enter you root password
    virtualenv -p python2.7 ~/sifra_venv
    source ~/sifra_venv/bin/activate

Note, in the above, the first command `sudo apt-get -y build-dep matplotlib` installs all the build dependencies for matplotlib.

Once inside the `virtualenv`, navigate to the `sifra` code:
    
    cd sifra # and not cd sifra/sifra. This is where the requirements.txt exists
    pip install -r requirements.txt

#### How to run the SIRA code

Running the sifra code is simple. First
    
    cd sifra # and not cd sifra/sifra

Run the `sifra` code as
    
    python -m sifra simulation_setup/config_ps_X.conf

#### Run tests
To run tests use either `nose` or `unittest` like the following from the first level `sifra` directory:
    
    cd sifra  # and not cd sifra/sifra
    python -m unittest discover tests
    or
    nosetest
