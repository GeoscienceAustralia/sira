See the full [documentation on GitHub Pages](http://geoscienceaustralia.github.io/sifra/index.html).


# What is it?

SIFRA is a **System for Infrastructure Facility Resilience Analysis**.
It comprises a method and software tools that provide a framework
for simulating the fragility of infrastructure facilities to natural
hazards, based on assessment of the fragilities and configuration of
components that comprises the facility. Currently the system is
designed to work with earthquake hazards only.

The following are some key features of this tool:

- Written in Python: It is written in Python, and there is no
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

### Using Anaconda

We recommend using ``conda`` for managing virtual environments and
packages required for running ``sifra``.

For the sake of simplicity, we recommend using ``Anaconda``. It is a
free Python distribution, and comes with the ``conda`` tool which is
both a package manager and environment manager. Instructions for
installing ``Anaconda`` are here:
<http://docs.continuum.io/anaconda/install>

**Prerequisites:** You will need to install ``Graphviz`` for the
drawing the system diagram through networkx and pygraphviz.
Please visit: <http://www.graphviz.org/>
and download the appropriate version for your operating system.
Please follow the posted download instructions carefully.
After installation you may need to update the PATH variable
with the location of the Graphviz binaries.

For windows systems you will need to install
Microsoft Visual C++ Compiler for Python 2.7:
<http://aka.ms/vcpython27>

On Ubuntu you can

```
apt-get update && apt-get install -y \
    build-essential pkg-config \
    graphviz libgraphviz-dev \
    xml2 libxml2-dev
```

Some packages we need are not hosted in the main ``conda`` package
repository. In such cases we will host them in our own user channel.
We suggest adding the following channels to the default::

    $ conda config --add channels https://conda.anaconda.org/anaconda
    $ conda config --add channels https://conda.anaconda.org/marufr

Run the following command to confirm the additional channels have
been added:

    $ conda config --get channels

**For OS X and Linux-64 systems:** It should be possible to set up a
full run environment solely through the \*.yml environment specification
file. For OS X run the following commands:

    $ conda env create -f environment_osx.yml
    $ source activate sifra_env

For Linux-64 systems, the commands are identical, you will just need
to use the environment specification file for Linux.

**For Windows systems**, a similar process needs to be followed, with
some exceptions. First run:

    $ conda env create -f environment_win64.yml
    $ activate sifra_env

This will install most requirements except for ``igraph`` and
``pygraphviz``. Compiling these packages under windows can be very
challenging. The simplest and most reliable options is to download
the the appropriate wheels from Christoph Gohlke's unofficial page
of Windows binaries:
<http://www.lfd.uci.edu/~gohlke/pythonlibs/>

For Windows 64 bit systems, you will need to download the ``wheels`` for
[python-igraph](http://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph)
and [pygraphviz](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pygraphviz):

- ``python_igraph-0.7.1.post6-cp27-none-win_amd64.whl``
- ``pygraphviz-1.3.1-cp27-none-win_amd64.whl``

Install these downloaded ``wheels`` with pip:

    $ pip install <pkg_name>.whl



### Using Docker

If you have Docker installed, you can build a container for working with
sifra by running the command

```
docker build -t sifra .
```

The primary advantage of working with docker is that you do not have to worry
abouti setting up the python environment, which is done when building the
container and isolated from your own environment.

To run an interactive container you can use:

```
docker run -it -v "$(pwd):/sifra" --name sifra sifra
```

This will give you a terminal inside the container in which you can execute
commands. Inside the container you can find the current directory mapped at
`/sifra`. You can modify files either within the container or the host and the
changes will be available in both.

Alternatively, you might want a container running in the background which you
can execute commands at (using
[docker exec](https://docs.docker.com/engine/reference/commandline/exec/)). In
this case you would start the container with:

```
docker run -id -v "$(pwd):/sifra" --name sifra sifra
```

One could then, for example, run the unit tests for the modelling package with:

```
docker exec sifra python -m unittest sifra.modelling.test_structural
```

In any case, once you are done you should destroy the container with

```
docker kill sifra
docker rm sifra
```

or, if your too lazy to type two lines...

```
docker rm -f sifra
```

Several other containers are provided to help with development. These are
defined in the other *Dockerfile*s in the present directory, and are:

- *Dockerfile-api*: Provides a web API which is used for parameterising
model components (at this stage just response functions) and serialising them.
This is presently (at Feb 2017) a prototype and provides only a small subset
of what we hope for.

- *Dockerfile-gui-dev*: Provides an [Agular2](https://angular.io/) application for
defining model components built on top of the API mentioned above. The application
is hosted using Angular's development server and can be accessed on *localhost:4200*.

- *Dockerfile-gui-prod*: For deploying the web application in production. This
does a production build of the Angular project and hosts it using
[busybox](https://www.busybox.net/). The app is still exposed on port 4200, so
to host it at port 80 one would start it with:

  ```
  docker build -t sifra-gui -f Dockerfile-gui-prod .
  ```

  and start it with (for example):

  ```
  docker run -d -p 80:4200 --restart always sifra-gui-prod
  ```

#### Docker Compose

By far the easiest way to run the system for development is with
[docker-compose](https://docs.docker.com/compose/), which can be done with:

```
docker-compose up
```

Assuming that you start the system this way in the current folder, you can:

- attach to the sifa image to run models and tests with

  ```
  docker attach sifra_sifra_1
  ```

- access the GUI for defining fragility functions at *http://localhost:4200*, and

- access the web API at *http://localhost:5000*.

The both the API and GUI will stay in sync with your code.

You can tear the system down (destroying the containers) with

```
docker-compose down
```


## Running the Code

Clone the repository onto your system. Detailed instructions can
be found [here](https://help.github.com/articles/cloning-a-repository/).

    $ git clone https://github.com/GeoscienceAustralia/sifra.git sifra

Move into the root directory for the ``SIFRA`` code:

    $ cd sifra    # NOT cd sifra/sifra

Run the `sifra` code as a module, with the requisite configuration
file:

    $ python -m sifra simulation_setup/config_file.conf

Depending on the scale of the model, and simulation paramters chosen,
it may take between a few minutes and a few days to complete a run.

To fit a system fragility for the facility to the simulated data
generated in the previous step, and a simple normal restoration
model, run the command:

    $ python sifra/fit_model.py simulation_setup/config_file.conf

To simulate the `component type` loss analysis, restoration prognosis,
and generate the component criticality plot, run the command:

    $ python sifra/scenario_loss_analysis.py simulation_setup/config_file.conf


## Running the Tests

To run tests use either ``nose`` or ``unittest``.
Example (from the first level 'sifra' directory):

    $ cd sifra  # and not cd sifra/sifra
    $ python -m unittest discover tests

or, simply run:

    $ nosetest

If you are using docker as described above, you can do this within the sifra
container.

:grey_exclamation: NOTE: Project needs a more comprehensive test suite.

## Todo

- Restructure of Python code. While the modularisation is not too bad (each
  module is probably close to sensible), the names of modules are terrible.

- The handling of types within the web API is inconsistent; in some cases it
  works with instances, in others dicts and in others, JSON docs. This
  inconsistency goes beyond just the web API and makes everything harder to get.
  One of the main reasons for this is the late addtion of 'attributes'. These
  are meant to provide metadata about instances and I did not have a clear
  feel for whether they should be part of the instance or just associated with
  it. I went for the latter, which I think is the right choice, but did not
  have the time to make the API consistent throughout.

- Much work needs to be done on the GUI. It is currently **horrible**. The
  Angular2 code contained herein is my first experience with it and being a
  prototype with a small time budget, I did not:
  - spend much time being idiomatically consistent,
  - leveraging existing elements of Angular2 (like
    [reacitve forms](https://angular.io/docs/ts/latest/guide/reactive-forms.html),
  - ... writing tests.

- Consider whether a framework like [Redux](http://redux.js.org/) would be useful.

- Perhaps get rid of ng\_select. I started with this before realising how easy
  simple HTML selects would be to work with and before reading about reacitve
  forms (I'm not sure how/if one could use ng\_select with them). One benefit of
  ng\_select may be handling large lists and one may want to do some testing
  before removing it.

- Move the logic of merging an instance with its metadata (currently handled in
  sifra.structural.\_merge\_data\_and\_metadata) to the javascript. The document
  produced by that method is heavy due to its repetativeness and would hence be
  slow to pass around over the net. The logic is straight forward and would be
  easy to implment in javascript given the 'metadata' and instance.


## Special files

*sifra/components.csv*: Used to populate the components categories table. All unique
  rows in the table are used to form the set of possible component types. This
  is loaded from *sifra/components.py* if the (sqlite) database *db/sqlite.db*
  does not exist. If some other DB were used, the logic to choose when to load
  this data would need to be chosen and implemented to suite.

