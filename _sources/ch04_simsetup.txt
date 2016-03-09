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
  Description: Unique id for component in system. This is an instance
  of `component_type` |br|
  Data Type: String. It is recommended to use alphanumeric characters,
  starting with a letter, and logically distinct parts of the name
  separated by underscores |br|
  Example: 'stack_1' |br|


`component_type`
  Description:  The general type of a component. Represents a broad
                category of equipment. |br|
  Data Type:    String. It is recommended to use alphanumeric characters,
                starting with a letter, and logically distinct parts of
                the name separated by underscores |br|
  Example:      'Stack' |br|


`component_class`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`cost_fraction`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`node_type`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`node_cluster`
  Description: |br|
  Data Type: |br|
  Example: |br|


`op_capacity`
  Description: |br|
  Data Type: |br|
  Example: |br|


Connections between Components: *component_connections*
-------------------------------------------------------

`Origin`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`Destination`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`Capacity`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`Weight`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`Distance`
  Description: |br|
  Data Type: |br|
  Example:  |br|


Configuration of Output Nodes: *output_setup*
---------------------------------------------

`OutputNode`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`ProductionNode`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`Capacity`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`CapFraction`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`Priority`
  Description: |br|
  Data Type: |br|
  Example:  |br|


Configuration of Supply Nodes: *supply_setup*
---------------------------------------------

`InputNode`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`Capacity`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`CapFraction`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`CommodityType`
  Description: |br|
  Data Type: |br|
  Example:  |br|


Damage Algorithms for Component Types: *comp_type_dmg_algo*
-----------------------------------------------------------

`component_type`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`damage_state`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`damage_function`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`mode`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`damage_median`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`damage_logstd`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`damage_ratio`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`functionality`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`minimum`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`sigma_1`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`sigma_2`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`recovery_mean`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`recovery_std`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`recovery_95percentile`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`fragility_source`
  Description: |br|
  Data Type: |br|
  Example:  |br|


Definition of Damage States: *damage_state_def*
-----------------------------------------------

`component_type`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`damage_state`
  Description: |br|
  Data Type: |br|
  Example:  |br|


`damage_state_definitions`
  Description: |br|
  Data Type: |br|
  Example:  |br|

