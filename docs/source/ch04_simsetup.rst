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
    ├── LICENSE                   <-- License file: Apache 2
    ├── README.md                 <-- Summary documentation
    ├── data
    │   └── ps_coal               <-- Directory for specified facility type
    │       ├── input             <-- Facility definition file resides here
    │       └── output            <-- The simulation results are here
    │           └── raw_output    <-- Raw data/binary outputs for provenance
    ├── docs
    │   └── source                <-- ReST files for Sphinx documentation
    │       ├── _static
    │       └── _templates
    ├── sifra                     <-- The MODULES/SCRIPTS
    ├── simulation_setup          <-- Scenario config files
    └── tests                     <-- Test scripts


.. _scenario-config-file:

Scenario Definition File
========================

The simulation 'scenario' definition file is located in the following directory 
(relative to the root dir of source code)::

    ./simulation_setup

The following table lists the parameters in the config file, their
description, and representative values.

.. csv-table::
   :header-rows: 1
   :widths: 25, 20, 55
   :stub-columns: 1
   :file: _static/files/scenario_config_parameters.csv


.. _facility-config-file:

Facility Definition File
========================

The file that defines a facility is located in the following directory
(relative to the root dir of source code)::

    ./data/<facility_type>/input
