.. SIFRA documentation master file, created by
   sphinx-quickstart on Thu Feb 25 09:28:33 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SIFRA documentation
*******************

https://github.com/GeoscienceAustralia/sifra | **Release:** |release|

SIFRA is a **System for Infrastructure Facility Resilience Analysis**.
It comprises a method and software tools that provide a framework
for simulating the fragility of infrastructure facilities to natural
hazards, based on assessment of the fragilities and configuration of
components that comprises the facility. Currently the system is
designed to work with earthquake hazards only. SIFRA was developed in
`Geoscience Australia (GA) <http://www.ga.gov.au/>`_ in support of the
agency's vision to contribute to enhancing the resilience of communities
in Australia and its region.

Feature Highlights
==================

- **Written in Python:** |br|
  It is written in Python, and there is no
  dependency on proprietary tools. It should run on OS X, Windows, and
  Linux platforms.

- **Flexible Facility Model:** |br|
  :term:`Facility` data model is based on network theory, allowing
  the user to develop arbitrarily complex custom facility models
  for simulation.

- **Extensible Component Library:** |br|
  User can define new instances of `component_type`
  (the building blocks of a facility) and link it to existing or
  custom fragility algorithms.

- **Component Criticality Analysis:** |br|
  Scenario Analysis tools allow users to identify the cost of
  restoration for chosen scenarios, expected restoration times,
  and which component upgrades can most benefit the system.

- **Restoration Prognosis:** |br|
  User can experiment with different levels of hazards and
  post-disaster resource allocation to gauge restoration
  times for facility operations.

Contents
========

.. include:: chapters_for_toc.txt

.. toctree::

    bibliography
    reportglossary


.. raw:: html

    <div style="width: 100%; height: 1px;
    background: #dddddd;
    margin: 2em 0 2em 0; overflow: hidden;"> </div>

.. include:: copyrightnotice.rst
