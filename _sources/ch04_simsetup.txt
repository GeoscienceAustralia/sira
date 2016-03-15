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


.. _input__component_list:

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

  :Example:     'Emission Management' -- stacks and ash disposal systems
                belong to different typologies, but both contribute to
                the function of emission management.


`cost_fraction`
  :Description: Value of the component instance a fraction of the
                total system cost, with the total cost being 1.0

  :Data Type:   Float.
                :math:`{\{x \in \mathbb{R} \mid 0 \le x \le 1\}}`

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

  :Data Type:   Float.
                :math:`{\{x \in \mathbb{R} \mid 0 \leq x \leq 1\}}`

  :Example:     1.0 (default value)


.. _input__component_connections:

Connections between Components: *component_connections*
-------------------------------------------------------

`Origin`
  :Description: The node (component) to which the tail of a
                directional edge is connected. For undirected graphs
                the origin/destination designation is immaterial.
                For bidirectional connections, you will need to define
                two edges, e.g. A |rightarrow| B, and B |rightarrow| A

  :Data Type:   String. Must be one of the entries in the
                `component_id` columns in the `component_list` table.

  :Example:     'stack_1'


`Destination`
  :Description: The node (component) on which the head of a
                directional edge terminates. For undirected graphs
                the origin/destination designation is immaterial.

  :Data Type:   String. Must be one of the entries in the
                `component_id` columns in the `component_list` table.

  :Example:     'turbine_condenser_1'


`Capacity`
  :Description: Capacity of the edge.
                It can be more than the required flow.

  :Data Type:   Float.
                :math:`{\{x \in \mathbb{R}\ \mid \ 0 \leq x \leq 1\}}`

  :Example:     1.0 (default value)


`Weight`
  :Description: This parameter can be used to prioritise an edge or
                a series of edges (a path) over another edge or set
                of edges.

  :Data Type:   Integer

  :Example:     1 (default value)


.. _input__output_setup:

Configuration of Output Nodes: *output_setup*
---------------------------------------------

`OutputNode`
  :Description: These are the 'sink' nodes representing the load or
                the aggregate consumer of the product(s) of the system.
                These are not real components, but a modelling construct.
                These nodes are not considered in the fragility
                calculations.

  :Data Type:   String. Must be one of the entries in the
                `component_id` columns in the `component_list` table,
                and must be of `node_type` sink.

  :Example:     'output_1'


`ProductionNode`
  :Description: These are the real terminal nodes within the facility
                system model. The completed 'product' of a system exits
                from this node.

  :Data Type:   String. Must be one of the entries in the
                `component_id` columns in the `component_list` table,
                and must be of `node_type` transshipment.

  :Example:     'gen_1'


`Capacity`
  :Description: Production capacity that the specific production node
                is responsible for. The unit depends on the
                type of product the system produces
                (e.g. MW for generator plant).

  :Data Type:   Float

  :Example:     300


`CapFraction`
  :Description: The fraction of total production capacity of the
                output nodes. The sum of capacities of all nodes must
                equal 1.0.

  :Data Type:   Float :math:`{\{x \in \mathbb{R} \mid 0 < x \leq 1\}}`

  :Example:     0.5


`Priority`
  :Description: This parameter is used to assign relative sequential
                priority for output/production nodes in for the
                purposes of post-disaster recovery

  :Data Type:   Integer.
                :math:`{\{x \in \mathbb{Z} \mid 1 \leq x \leq n\}}`,
                where `n` is the total number of output nodes

  :Example:     _


.. _input__supply_setup:

Configuration of Supply Nodes: *supply_setup*
---------------------------------------------

`InputNode`
  :Description: The `component_id` of the input node.

  :Data Type:   String. Must be one of the entries in the
                `component_id` columns in the `component_list` table,
                and its `node_type` must be `supply`.

  :Example:     'coal_supply'


`Capacity`
  :Description: The operational capacity of the node. It can be a real value
                value if known, or default to 100%.

  :Data Type:   Float.
                :math:`{\{x \in \mathbb{R} \mid 0.0 \lt x \leq 100.0\}}`

  :Example:     100.0 (default value)


`CapFraction`
  :Description: What decimal fractional value of the input commodity
                enters the system through this input node.

  :Data Type:   Float.
                :math:`{\{x \in \mathbb{R} \mid 0.0 \lt x \leq 1.0\}}`

  :Example:     1.0


`CommodityType`
  :Description: The type of commodity entering into the system through
                the specified input node.

  :Data Type:   String.

  :Example:     For a coal-fired power station there might be two
                commodities, namely coal and water.
                For an electric substation it is electricity.
                For a water treatment plant, it is waster water.


.. _input__comp_type_dmg_algo:

Damage Algorithms for Component Types: *comp_type_dmg_algo*
-----------------------------------------------------------

`component_type`
  :Description: The type of component, based on the typology definitions
                being used in the system model.

  :Data Type:   Alphanumeric characters.
                May use dashes '-' or underscores '_'.
                Avoid using special characters.

  :Example:     'Demineralisation Plant'


`damage_state`
  :Description: The list of damage states used in defining the
                damage scale being modelled within the system.

  :Data Type:   String. Fixed, pre-determined state names.

  :Example:     For a four-state sequential damage scale,
                the following damage states are used:

                1. DS1 Slight
                2. DS2 Moderate
                3. DS3 Extensive
                4. DS4 Complete


`damage_function`
  :Description: The probability distribution for the damage function.
                Currently only log-normal curves are used, but additional
                distributions can be added as required.

  :Data Type:   String.

  :Example:     'lognormal'


`mode`
  :Description: Number indicating the mode of the function.
                Currently can handle only unimodal or bimodal functions.

  :Data Type:   Integer [1,2]

  :Example:     1


`damage_median`
  :Description: Median of the damage function.
                A median will need to be defined for each damage state.
                It should be typically be progressively higher for more
                severe damage states:
                :math:`{\mu_{DS1} \leq \mu_{DS2} \leq \mu_{DS3} \leq \mu_{DS4}}`

  :Data Type:   Float.

  :Example:     0.45


`damage_logstd`
  :Description: Standard deviation of the damage function.
                It will need to be defined for each damage state.
                The value of standard deviation should be such that
                the curves do not overlap.

  :Data Type:   Float.

  :Example:     0.20


`damage_ratio`
  :Description: The fractional loss of a component's value for damage
                sustained at a given damage state. This parameter links
                a damage state to expected direct loss of component value.

  :Data Type:   Float.
                :math:`{\{x \in \mathbb{R} \mid 0.0 \leq x\}}`.
                A value of 0 indicates no loss of value, and
                a value of 1.0 indicates complete loss.
                In special cases the the value of loss ratio can be
                greater than 1.0, which indicates complete loss of
                component and additional cost of removal, disposal, or
                securing or destroyed component.

  :Example:     Damage ratio of 0.30 for damage state "DS2 Moderate"


`functionality`
  :Description: An unitless fractional value indicating the functional
                capacity of a component for a given damage state.
                This parameter links damage states to expected
                post-impact residual functionality of the component.

  :Data Type:   Float.
                :math:`{\{x \in \mathbb{R} \mid 0.0 \leq x \leq 1.0\}}`.
                A value of 0 indicates no loss of value, and
                a value of 1.0 indicates complete loss.
                In special cases the the value of loss ratio can be
                greater than 1.0, which indicates complete loss of
                component and additional cost of removal, disposal, or
                securing or destroyed component.

  :Example:     A stack of a thermal power station is expected to remain
                fully functional (functionality==1), under 'Slight'
                damage state, i.e. under conditions of minor damage to
                structure with deformation of holding down bolts and with
                some bracing connections.


`minimum`
  :Description: Minimum value for which the damage algorithm is
                applicable.

  :Data Type:   Float.

  :Example:     The algorithms presented by Anagnos :cite:`Anagnos1999`
                for 500kV circuit breakers are only applicable for
                PGA values of 0.15g and above, for the various noted
                failure modes.


`sigma_1`
  :Description: The first standard deviation for a bimodal
                damage function.

  :Data Type:   Float, for a bimodal function. However, for
                single mode functions, use 'NA'.

  :Example:     _


`sigma_2`
  :Description: The second standard deviation for a bimodal
                damage function.

  :Data Type:   Float, for a bimodal function. However, for
                single mode functions, use 'NA'.

  :Example:     _


`recovery_mean`
  :Description: The mean of the recovery function. Component and
                system restoration time are assumed to follow the
                normal distribution.

  :Data Type:   Float.

  :Example:     _


`recovery_std`
  :Description: The standard deviation of the recovery function.
                Component and system restoration time are assumed
                to follow the normal distribution.

  :Data Type:   Float.

  :Example:     _


`recovery_95percentile`
  :Description: Some times it is difficult to get the concept of
                standard deviation across to an audience of
                infrastructure experts, and hence it is difficult
                to get a reliable value for it. In such cases we can
                obtain a 95th percentile value for recovery time, and
                translate that to standard deviation for a normal
                distribution using the following equation:

                .. math::

                    \begin{align}
                    &X_{0.95} = \mu + Z_{0.95} \sigma \\
                    \Rightarrow &X_{0.95} = \mu + \Phi^{-1}(0.95) \sigma \\
                    \Rightarrow &\sigma = \frac{X_{0.95} - \mu}{\Phi^{-1}(0.95)}
                    \end{align}

  :Data Type:   Float

  :Example:     _


`fragility_source`
  :Description: Which source the fragility algorithm was adopted from,
                how it was adapted, or how it was developed.

  :Data Type:   Free text

  :Example:     _


.. _input__damage_state_def:

Definition of Damage States: *damage_state_def*
-----------------------------------------------

This table documents the physical damage characteristics that are implied
by the damage states used to model the fragility of the system components.


`component_type`
  The entries here are the same as noted under
  `component_type` in the :ref:`Componenet Type Damage
  Algorithm table table <input__comp_type_dmg_algo>`.


`damage_state`
  The entries here are the same as noted under
  `damage_state` in the :ref:`Component Type Damage
  Algorithm table table <input__comp_type_dmg_algo>`.


`damage_state_definitions`
  :Description: The physical damage descriptors corresponding
                to the damage states.

  :Data Type:   Free text.

  :Example:     230 kV Current Transformers would be said to be in
                `Failure` state if there is
                "porcelain cracking, or overturning."

