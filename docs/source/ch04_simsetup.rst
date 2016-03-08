.. _simulation-inputs:

****************
Simulation Setup
****************

Setting a simulation requires populating two different sets of inputs:

- Scenario configuration
- Facility system configuration

These two sets of input data are contained in two separate files. These files,
their parameters, data layout, and sample input data are presented in the
remainder of this Section. In the course of the discussion it should be useful
to the keep the directory structure of the code in mind::

    .
    ├── LICENSE                  <-- License file: Apache 2
    ├── README.md                <-- Summary documentation
    │
    ├── docs                     <-- Sphinx documentation files
    │   └── source
    │       ├── _static
    │       ├── _templates
    │       └── extensions
    │
    ├── models                   <-- Facility system models reside here
    │   └── <facility_type>      <-- Dir for models of specified type
    │
    ├── output                   <-- Simulation results are stored here
    │   └── scenario_X           <-- Simulation project name
    │       └── raw_output       <-- Raw data/binary outputs for provenance &
    │                                  post-processing
    ├── sifra                    <-- The MODULES/SCRIPTS
    ├── simulation_setup         <-- Scenario config files
    └── tests                    <-- Test scripts


.. _scenario-config-file:

Scenario Definition File
========================

The simulation 'scenario' definition file is located in the following directory
(relative to the root dir of source code)::

    ./simulation_setup/

The following table lists the parameters in the config file, their
description, and representative values.

.. csv-table::
   :header-rows: 1
   :widths: 20, 20, 60
   :stub-columns: 0
   :file: _static/files/scenario_config_parameters.csv


.. _facility-config-file:

Facility Definition File
========================

The system definition files for a facility of type ``<facility_type_A>``
is located in the following directory (relative to the root dir of
source code)::

    ./models/<facility_type_A>/

The system model is defined using an MS Excel spreadsheet file.
It contains five worksheets. The names of the worksheets are fixed.
The function and format of these worksheets are described in the
following subsections:


List of Component: *component_list*
-----------------------------------

The *component_list* has the following parameters:

`component_id`
 |**Description:** unique id for component in system
 |**Data Type:** alphanumeric, parts of the name can be separated by underscores
 |**Example:** stack_1


`component_type`
|  Description:
|  Data Type:
|  Example: stack_1


`component_class`
| Description:
| Data Type:
| Example: stack_1


`cost_fraction`
| Description:
| Data Type:
| Example: stack_1


`node_type`
| Description:
| Data Type:
| Example: stack_1


`node_cluster`
|Description:
|Data Type:
|Example: stack_1


`op_capacity`
|Description:
|Data Type:
|Example: stack_1


Connections between Components: *component_connections*
-------------------------------------------------------

``Origin``
``Destination``
``Capacity``
``Weight``
``Distance``


Configuration of Output Nodes: *output_setup*
---------------------------------------------

``OutputNode``
``ProductionNode``
``Capacity``
``CapFraction``
``Priority``


Configuration of Supply Nodes: *supply_setup*
---------------------------------------------

``InputNode``
``Capacity``
``CapFraction``
``CommodityType``


Damage Algorithms for Component Types: *comp_type_dmg_algo*
-----------------------------------------------------------

``component_type``
``damage_state``
``damage_function``
``mode``
``damage_median``
``damage_logstd``
``damage_ratio``
``functionality``
``minimum``
``sigma_1``
``sigma_2``
``recovery_mean``
``recovery_std``
``recovery_95percentile``
``fragility_source``


Definition of Damage States: *damage_state_def*
-----------------------------------------------

``component_type``
``damage_state``
``damage_state_definitions``

