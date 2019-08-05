.. _simulation-inputs:

**************************
Simulation and Model Setup
**************************

Setting a simulation requires populating two different sets of inputs:

- Simulation scenario configuration
- Infrastructure model configuration

These two sets of input data are contained in two separate files. These files,
their parameters, data layout, and sample input data are presented in the
remainder of this Section. 

*Naming Requirement of Input File*:
The model file name must begin or end with the term 'model'. 
Similarly, the config file must have term 'config' at the beginning or
end of its name. 

*Location Requirement for Input Files*:
The input files must reside within a directory named `input`. The parent
directory is named after the infrastructure asset (by convention) and may
be located in any storage location where the user has access. The code is
notified of the location with a flag `-d` at execution time.

The outputs generated from the simulation are stored in a directory called
`output` within the same parent directory.

Within the defined model directory, there must exist at least one directory
with two files:

- the config file
- the model file

The directory structure of the code is as follows::

    .
    ├── docs                        <-- Sphinx documentation files
    │   └── source
    ├── hazard                      <-- Hazard scenario files (for networks)
    ├── installation                <-- Installation scripts for dev envs
    ├── scripts
    ├── sira                        <-- The core code reside here
    │   └── modelling
    ├── tests                       <-- Test scripts + data for sanity checks
    │   ├── historical_data
    │   ├── models
    │   └── simulation_setup
    │
    ├── LICENSE                      <-- License file
    ├── README.md                    <-- Summary documentation
    ├── __main__.py                  <-- Entry point for running the code
    ├── LICENSE
    └── README.adoc                  <-- Basic documentation and installation notes

.. _simulation-setup-file:

Simulation Setup File
=====================

The code needs a setup file for configuring the model and simulation scenario.
The code expects it in JSON format. 

The following table lists the parameters in the config file,
their description, and representative values.

.. include::
   ./_static/files/model_params__simulation_setup.txt


.. _model-definition-file:

Infrastructure Model Definition File
====================================

The system model is defined using an MS Excel spreadsheet file.
It contains five worksheets. The names of the worksheets are fixed.
The function and format of these worksheets are described in the
following subsections:


.. _inputdata__component_list:

List of Components: component_list
----------------------------------

The *component_list* has the following parameters:

.. include::
   ./_static/files/model_params__component_list.txt


.. _inputdata__component_connections:

Connections between Components: component_connections
-----------------------------------------------------

.. include::
   ./_static/files/model_params__component_connections.txt


.. _inputdata__supply_setup:

Configuration of Supply Nodes: supply_setup
-------------------------------------------

.. include::
   ./_static/files/model_params__supply_setup.txt


.. _inputdata__output_setup:

Configuration of Output Nodes: output_setup
-------------------------------------------

.. include::
   ./_static/files/model_params__output_setup.txt


.. _inputdata__comp_type_dmg_algo:

Component Type Damage Algorithms: comp_type_dmg_algo
----------------------------------------------------

.. include::
   ./_static/files/model_params__comp_type_dmg_algo.txt


.. _inputdata__damage_state_def:

Definition of Damage States: damage_state_def
---------------------------------------------

This table documents the physical damage characteristics that are implied
by the damage states used to model the fragility of the system components.

.. include::
   ./_static/files/model_params__damage_state_def.txt

