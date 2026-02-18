.. _simulation-input-setup:

****************
Simulation Setup
****************

The simulation setup requires two different sets of inputs. 
These two sets of input data are contained in two separate files:

- The config file: this file sets up the simulation scenario parameters
- The model file: this defines the components, their connections, along 
  with the fragility, loss, and recovery parameters for the infrastructure
  model.

These files, their parameters, data layout, and sample input data are
presented in the remainder of this Section. 

*Naming Requirement of Input File*:
The model file name must begin or end with the term 'model'. 
Similarly, the config file must have the term 'config' at the beginning
or end of its name. 

*Location Requirement for Input Files*:
The input files must reside within a directory named `input`. The parent
directory is named after the infrastructure asset by convention, but it can
have any name allowed by the local file system. This dir can be located in
any storage location where the user has access. The code is informed of
the location with a flag `-d` at execution time.

The outputs generated from the simulation are stored in a directory called
`output` within the same parent directory.

For illustration purposed we assume a hypothetical project "PROJECT HAN",
with the project folder located in the root directory. We also assume
that within the project, we are modelling two systems named GISKARD and
DANEEL. For this given setup, input-output dir structure for the project
will be as follows::

    .
    └── <PROJECT_HAN>
        ├── <SYSTEM_GISKARD>
        │   ├── input
        │   │   ├── config_system_GR.json
        │   │   └── model_system_GR.json
        │   └── output
        │       ├── ...
        │       ├── ...
        │       └── ...
        │
        └── <SYSTEM_DANEEL>
            ├── input
            │   ├── config_system_DO.json
            │   └── model_system_DO.json
            │
            └── output
                ├── ...
                ├── ...
                └── ...


.. _simulation-setup-file:

Simulation Configuration File
=============================

The code needs a setup file for configuring the simulation scenario.
The expected file format is JSON. The code *can* support any of three formats:
`ini`, `conf`, or `json`, though formats other than JSON are discouraged and
details of their implementation will not be discussed here.

The following table lists the parameters in the config file,
with a brief description and representative values.


Scenario Parameters
-------------------

.. include::
   ./_static/files/model_params__sim_config_scenario.txt


Hazard Parameters
-----------------

.. include::
   ./_static/files/model_params__sim_config_hazard.txt


Restoration Parameters
----------------------

.. include::
   ./_static/files/model_params__sim_config_restoration.txt


System Category Metadata Parameters
-----------------------------------

.. include::
   ./_static/files/model_params__sim_config_meta.txt


Simulation Configuration Flags
------------------------------

.. include::
   ./_static/files/model_params__sim_config_flags.txt
