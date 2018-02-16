See the full [documentation on GitHub Pages](http://geoscienceaustralia.github.io/sifra/index.html).


# What is it?

SIFRA is a **System for Infrastructure Facility Resilience Analysis**.
It comprises a method and software tools that provide a framework
for simulating the fragility of infrastructure facilities to natural
hazards, based on assessment of the fragilities and configuration of
components that comprises the facility. Currently the system is
designed to work with earthquake hazards only.

The following are some key features of this tool:

- Open Source: It is written in Python, and there is no
  dependency on proprietary tools. It should run on OS X, Windows, and
  Linux platforms.

- Flexible Facility Model: ``facility`` data model is based on network
  theory, allowing the user to develop arbitrarily complex custom
  facility models for simulation.

- Extensible Component Library: User can define new ``component types``
  (the building blocks of a facility) and link it to existing or
  custom fragility algorithms.

- Component Criticality: Scenario Analysis tools allow users to
  identify the cost of restoration for chosen scenarios, expected
  restoration times, and which component upgrades can most benefit
  the system.

- Restoration Prognosis: User can experiment with different levels of
  hazards and post-disaster resource allocation to gauge restoration
  times for facility operations.


We are currently extending the system to provide facilities for sharing models
and model "components" (we are still working on the nomenclature... hence the
quotes) such as infrastrure, response functions and entire system models. This
system is not integrated into the modelling system proper at present but will be
over the course of coming months (as at Feb 2017).


# Setup Instructions

It is good practice to set up a virtual environment for working with
developing code. This gives us the tools to manage the package
dependencies and requirements in a transparent manner, and impact of
dependency changes on software behaviour.

The virtual environments were initally described and implemented using
[Ananconda](https://www.continuum.io/). The new system is being designed as
microservices implemented in docker containers. If you have docker installed on
your system it is probably easiest to use the containers as described below.

## Building the Run Environment

The AWS and Docker configuration is now the preferred way to deploy and develop
 the application.

### Building an AMI for Dev Machines

To be able to have a easily replicable and deployable environment to run and 
test the code, we use AWS for provisioning dev machines. A basic dev instance can 
be setup using [Packer](https://www.packer.io/intro/), by running:

```
packer build build.json
```

in the current directory. The details, and options, for installation
are detailed in 

### Docker
The packer command that creates the AWS instance will run a 
shell script that will install Docker and clone the git repository
from Github. The details of the script are in `build-sifra-box.sh`

Creating the environment is done using the following commands:
```
cd sifra
docker build -t sifra .
```

To run an interactive container you can use:

```
docker run -it -v "$(pwd):/sifra" --name sifra sifra
```

This will give you a terminal inside the container in which you can execute
commands. Inside the container you can find the current directory mapped at
`/sifra`. You can modify files either within the container or the host and the
changes will be available in both.


For details of the commands see 
[Using docker](https://geoscienceaustralia.github.io/sifra/ch03_installation.html)
in the help documentation


## Running a simulation in Docker

First run an interactive container by using:

```
docker run -it -v "$(pwd):/sifra" --name sifra sifra
```

To run the sample scenario, while in in the /sifra directory run:
```
python -m sifra.infrastructure_response simulation_setup/test_scenario_ps_coal.conf
```

## Setting up a development Environment
Recent development has been done mostly on an AWS instance in PyCharm. This
requires tunnelling X11 through an SSH connection, which mostly works reasonably
well. 

The driver behind this is the authenticating proxy, which seems to break
docker in our use-case. Others have been able to run docker containers within
the GA network, but it was not considered a good use of development effort 
to attempt this with SIFRA.

PyCharm supports docker as detailed in the following links:

- [Pycharm Docker support](https://www.jetbrains.com/help/pycharm/docker.html)
- [Docker-Compose: Getting Flask up and running](https://blog.jetbrains.com/pycharm/2017/03/docker-compose-getting-flask-up-and-running/)

The following direcotories must be marked as 'Sources Root' in PyCharm. 

- sifra
- sifra-api

## Running the Code in Conda

Clone the repository onto your system. Detailed instructions can
be found [here](https://help.github.com/articles/cloning-a-repository/).

    $ git clone https://github.com/GeoscienceAustralia/sifra.git sifra

Move into the root directory for the ``SIFRA`` code:

    $ cd sifra    # NOT cd sifra/sifra

Run the `sifra` code as a module, with the requisite configuration
file:

    $ python -m  simulation_setup/config_file.conf

Depending on the scale of the model, and simulation parameters chosen,
it may take between a few minutes and a few days to complete a run.

To fit a system fragility for the facility to the simulated data
generated in the previous step, and a simple normal restoration
model, run the command:

    $ python sifra/fit_model.py simulation_setup/config_file.conf

To simulate the `component type` loss analysis, restoration prognosis,
and generate the component criticality plot, run the command:

    $ python sifra/scenario_loss_analysis.py simulation_setup/config_file.conf

## Testing

To run tests use either nose or unittest. Example (from the first level 
'sifra' directory):

    $ cd sifra/tests
    $ python -m unittest discover .

or, simply run:

    $ nosetest

Prior to running the tests you may need to delete the temporary database. 
Assuming you are in the root project directory:

    $ cd db
    $ rm sqlite.db 


:grey_exclamation: NOTE: We need to demonstrate code coverage of the tests. 


## Todo

- Restructure of Python code. While the simulation has been integrated with
  the json serialisation/deserialisation logic, the redundant classes should
  be removed and the capacity to create, edit and delete a scenario needs to 
  be developed.

- The handling of types within the web API is inconsistent; in some cases it
  works with instances, in others dicts and in others, JSON docs. This
  inconsistency goes beyond just the web API and makes everything harder to get.
  One of the main reasons for this is the late addtion of 'attributes'. These
  are meant to provide metadata about instances and I did not have a clear
  feel for whether they should be part of the instance or just associated with
  it. I went for the latter, which I think is the right choice, but did not
  have the time to make the API consistent throughout.

- Much work needs to be done on the GUI. It is currently very elementary. The
  Angular2 code contained herein is my first experience with it and being a
  prototype with a small time budget, I did not:
  - spend much time being idiomatically consistent,
  - leveraging existing elements of Angular2 (like
    [reactive forms](https://angular.io/docs/ts/latest/guide/reactive-forms.html),
  - ... writing tests.

- Consider whether a framework like [Redux](http://redux.js.org/) would be useful.

- Perhaps get rid of ng\_select. I started with this before realising how easy
  simple HTML selects would be to work with and before reading about reactive
  forms (I'm not sure how/if one could use ng\_select with them). One benefit of
  ng\_select may be handling large lists and one may want to do some testing
  before removing it.

- Move the logic of merging an instance with its metadata (currently handled in
  sifra.structural.\_merge\_data\_and\_metadata) to the javascript. The document
  produced by that method is heavy due to its repetativeness and would hence be
  slow to pass around over the net. The logic is straight forward and would be
  easy to implment in javascript given the 'metadata' and instance.

