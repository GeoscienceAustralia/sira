.. _model-setup:

**************************
Infrastructure Model Setup
**************************

The system model is defined using an MS Excel spreadsheet file.
It contains seven required worksheets, each representing modules that define
different aspect of the infrastructure model. The names of the worksheets
are fixed. The Excel file must be converted to JSON format before it can be
ingested by SIRA. The required script `convert_excel_files_to_json.py` is available in the `sira/tools <https://github.com/GeoscienceAustralia/sira/tree/master/sira/tools>`_ directory for
this purpose.

The format and function of the worksheets (i.e. model definition modules)
are described in the following subsections:

.. _inputdata__system_meta:

Basic System Metadata: system_meta
----------------------------------

.. include::
    ./_static/files/model_params__system_meta.txt

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

