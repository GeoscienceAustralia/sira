#### Setting Up the Environment

It is good practice to set up a virtual environment for working with developing code. This gives us the tools to manage the package dependencies and requirements in a transparent manner, and impact of dependency changes on software behaviour.

We recommend using `conda` for managing virtual environments and packages required for running `sifra`.

For the sake of simplicity, we recommend using `Anaconda`. It is a free Python distribution,
and comes with the `conda` tool which is both a package manager and environment manager.
Instructions for installing `Anaconda` are here: http://docs.continuum.io/anaconda/install

Some packages we need are not hosted in the main `conda` package repository. In such cases we will host them in our own user channel. We suggest adding the following channels to the default:

    conda config --add channels https://conda.anaconda.org/anaconda
    conda config --add channels https://conda.anaconda.org/marufr

Run the following command to confirm the additional channels have been added:

    conda config --get channels

Next, choose the the environment specification *.yml file relevant to your OS. for OS X run the following command:

    conda env create -f environment_osx.yml

Then acivate the newly created environment:

OS X, Linux:    `source activate sifra_env`

Windows:        `activate sifra_env`


#### Running SIFRA

First move into the root directory for the `SIFRA` code:
    
    cd sifra    # and not cd sifra/sifra

Run the `sifra` code as
    
    python -m sifra simulation_setup/config_file.conf

To fit a system fragility for the facility to the simulated data generated in the previous step, and a simple normal restoration model, run the command:

    python sifra/fit_model.py simulation_setup/config_file.conf

To simulate the `component type` loss analysis, restoration prognosis, and generate the component criticality plot, run the command:

    python sifra/scenario_loss_analysis.py simulation_setup/config_file.conf


#### Testing the Code

To run tests use either `nose` or `unittest`.
Example (from the first level `sifra` directory):
    
    cd sifra  # and not cd sifra/sifra
    python -m unittest discover tests

or, simply:

    nosetest

This project needs to be set up with a more comprehensive test suitew.
