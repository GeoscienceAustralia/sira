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
    ├── LICENSE                 <-- License file: Apache 2
    ├── README.md               <-- Summary documentation
    ├── data
    │   └── ps_coal             <-- Directory for specified facility type
    │       ├── input           <-- Facility definition file resides here
    │       └── output          <-- The simulation results are here
    │           └── raw_output  <-- Raw data/binary outputs for provenance
    ├── docs
    │   └── source              <-- ReST files for Sphinx documentation
    │       ├── _static
    │       └── _templates
    ├── sifra                   <-- The MODULES/SCRIPTS
    ├── simulation_setup        <-- Scenario config files
    └── tests                   <-- Test scripts


.. _scenario-config-file:

Scenario Definition File
========================

The simulation 'scenario' definition file is located in the following directory 
(relative to the root dir of source code)::

    ./simulation_setup

while the file that defines a facility resides under::

    ./data/<facility_type>/input


**Hazard Setup: **

PGA_MIN  = 0.0
PGA_MAX  = 1.5
PGA_STEP = 0.01
NUM_SAMPLES = 500
USE_ENDPOINT = True

HAZARD_TRANSFER_PARAM = 'PGA'   # EDP - Engineering Demand Parameter
HAZARD_TRANSFER_UNIT = 'g'     # EDP_unit - Demand Parameter Unit
SCENARIO_HAZARD_VALUES = [0.50]

**Restoration Setup: **

TIME_UNIT = 'week'
RESTORE_TIME_UPPER = 250.0
RESTORE_PCT_CHKPOINTS = 21
RESTORE_TIME_STEP = 1
RESTORE_TIME_MAX = 300.0

# The number of simultaneous components to work on.
# This represent resource application towards the restoration process.
RESTORATION_STREAMS = [5, 10, 20]

**System Setup: **

.. System Description & Configuration

SYSTEM_CLASSES = ["PowerStation", "Substation"]
SYSTEM_CLASS = "PowerStation"
SYSTEM_SUBCLASS = "Coal Fired"
PS_GEN_TECH = "Coal Fired"

COMMODITY_FLOW_TYPES = 2
SYS_CONF_FILE_NAME = 'sys_config_ps_sd0.35g.xlsx'

**Directory Specification: **

PROJECT_DIR = 'data/ps_coal'
INPUT_DIR_NAME = 'data/ps_coal/input'
OUTPUT_DIR_NAME = 'data/ps_coal/output'

**Test Switches: **

FIT_PE_DATA = True
FIT_RESTORATION_DATA = True
SAVE_VARS_NPY = True

**Test Switches: **
MULTIPROCESS = 1

**Test run or normal run" **
ENV = 1

