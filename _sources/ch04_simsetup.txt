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
    │   └── ps_coal             <-- Dir stating the facility type
    │       ├── input           <-- Facility definition file resides here
    │       └── output          <-- The simulation results are here
    │           └── raw_output  <-- Raw data/binary outputs for provenance
    ├── docs
    │   └── source              <-- Sphinx documentation ReST files
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

<**to be continued**>