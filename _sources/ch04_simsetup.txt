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

`HAZARD_TRANSFER_PARAM`
    :Description:   Engineering Demand Parameter

    :Data Type:     String

    :Example:       'PGA'


`HAZARD_TRANSFER_UNIT`
    :Description:   Demand Parameter Unit

    :Data Type:     String

    :Example:       'g'


`PGA_MIN`
    :Description:   Minimum value for PGA

    :Data Type:     Float

    :Example:       0.0


`PGA_MAX`
    :Description:   Maximum value for PGA

    :Data Type:     Float

    :Example:       1.5


`PGA_STEP`
    :Description:   Step size for incrementing PGA

    :Data Type:     Float

    :Example:       0.01


`NUM_SAMPLES`
    :Description:   Iterations for Monte Carlo process

    :Data Type:     Integer

    :Example:       500


`SCENARIO_HAZARD_VALUES`
    :Description:   The value(s) at which to assess facility response

    :Data Type:     List of floats

    :Example:       [0.50, 0.55]


`TIME_UNIT`
    :Description:   Unit of time for restoration time calculations

    :Data Type:     String

    :Example:       'week'


`RESTORE_PCT_CHKPOINTS`
    :Description:   Number of steps to assess functionality

    :Data Type:     Integer

    :Example:       21


`RESTORE_TIME_STEP`
    :Description:   Time increment for restoration period

    :Data Type:     Integer

    :Example:       1


`RESTORE_TIME_MAX`
    :Description:   Maximum value for restoration period assessment

    :Data Type:     Integer

    :Example:       300


`RESTORATION_STREAMS`
    :Description:   The number of simultaneous components to work on

    :Data Type:     List of integers

    :Example:       [5, 10, 20]


`SYSTEM_CLASSES`
    :Description:   The allowed facility system types

    :Data Type:     List of strings

    :Example:       ['PowerStation', 'Substation']


`SYSTEM_CLASS`
    :Description:   The facility system type to be modelled

    :Data Type:     String

    :Example:       'PowerStation'


`SYSTEM_SUBCLASS`
    :Description:   Sub-category of system

    :Data Type:     String

    :Example:       'Coal Fired'


`COMMODITY_FLOW_TYPES`
    :Description:   Number of input commodity types

    :Data Type:     Integer

    :Example:       2


`SYS_CONF_FILE_NAME`
    :Description:   File name for system config and fragility info

    :Data Type:     String

    :Example:       'sys_config_ps.xlsx'


`INPUT_DIR_NAME`
    :Description:   File path relative to code root

    :Data Type:     String

    :Example:       'data/ps_coal/input'


`OUTPUT_DIR_NAME`
    :Description:   File path relative to code root

    :Data Type:     String

    :Example:       'data/ps_coal/output'


`FIT_PE_DATA`
    :Description:   Flag for fitting Prob of Exceedance data

    :Data Type:     Boolean

    :Example:       True


`FIT_RESTORATION_DATA`
    :Description:   Fit model to simulated restoration data

    :Data Type:     Boolean

    :Example:       True


`SAVE_VARS_NPY`
    :Description:   Switch to indicate whether to save simulated
                    values in binary numpy format

    :Data Type:     Boolean

    :Example:       True


`MULTIPROCESS`
    :Description:   Switch to indicate whether to use multi-core processing.
                    0 -> False, 1 -> True

    :Data Type:     Integer

    :Example:       1


`RUN_CONTEXT`
    :Description:   Switch to indicate whether to run a full simulation,
                    or run test code. 0 -> run tests, 1 -> normal run.

    :Data Type:     Integer

    :Example:       1


.. .. csv-table::
   :header-rows: 1
   :widths: 30, 70
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
  :Description: Unique id for component in system. This is an instance
                of `component_type`

  :Data Type:   String.
                It is recommended to use alphanumeric characters,
                starting with a letter, and logically distinct parts
                of the name separated by underscores

  :Example:     'stack_1'


`component_type`
  :Description: The :term:`typology` of a system component.
                Represents a broad category of equipment.

  :Data Type:   String.
                It is recommended to use alphanumeric characters,
                starting with a letter, and logically distinct
                parts of the name separated by spaces.

  :Example:     'Stack'


`component_class`
  :Description: The general category of equipment. A number of
                component types can be grouped under this, e.g.
                'Power Transformer 100MVA 230/69' and
                'Power Transformer 50MVA 230/69' are both under
                the same component_class of 'Power Transformer'

  :Data Type:   String.
                It is recommended to use alphanumeric characters,
                starting with a letter, and logically distinct
                parts of the name separated by spaces.

  :Example:     'Power Transformer'


`cost_fraction`
  :Description: Value of the component instance a fraction of the
                total system cost, with the total cost being 1.0

  :Data Type:   Float :math:`{x \in \mathbb{R} | 0 \ge x \ge 1}`

  :Example:     0.03


`node_type`
  :Description: This indicates the role of the node (component) within
                network representing the system. For details, see
                `Facility System Model <_facility-system-model>`_

  :Data Type:   String.
                Must be one of four values:
                supply, transshipment, dependency, sink

  :Example:     'supply'


`node_cluster`
  :Description: This is an optional parameter to assist is drawing
                the system diagram. It indicates how the different
                component instances should be grouped together.

  :Data Type:   String

  :Example:     'Boiler System'


`op_capacity`
  :Description: Operational capacity of the component.
                One (1.0) indicates full functionality, and
                zero (0.0) indicates complete loss of functionality.
                Typically at the start of the simulation all components
                would have a value of 1.0.

  :Data Type:   Float :math:`{x \in \mathbb{R} | 0 \ge x \ge 1}`

  :Example:     1.0


Connections between Components: *component_connections*
-------------------------------------------------------

`Origin`
  :Description:

  :Data Type:

  :Example:


`Destination`
  :Description:

  :Data Type:

  :Example:


`Capacity`
  :Description:

  :Data Type:

  :Example:


`Weight`
  :Description:

  :Data Type:

  :Example:


`Distance`
  :Description:

  :Data Type:

  :Example:


Configuration of Output Nodes: *output_setup*
---------------------------------------------

`OutputNode`
  :Description:

  :Data Type:

  :Example:


`ProductionNode`
  :Description:

  :Data Type:

  :Example:


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

