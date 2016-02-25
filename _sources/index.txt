.. SIFRA documentation master file, created by
   sphinx-quickstart on Thu Feb 25 09:28:33 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SIFRA documentation
=================================

SIFRA stands for **System for Infrastructure Facility Resilience
Analysis**.
SIFRA comprises a method and software tools that provide a framework
for simulating the fragility of infrastructure facilities to natural
hazards, based on assessment of the fragilities and configuration of
components that comprises the facility. Currently the system is
designed to work with earthquake hazards only.

The following are some key features of this tool:

- Written in Python: It is written in Python, and there is no
  dependency on proprietary tools. It should run on OS X, Windows, and
  Linux platforms.
- Flexible Facility Model: ``facility`` data model is based on network
  theory, allowing the user to develop arbitrarily complex custom
  facility models for simulation.
- Extensible Component Library: User can define new ``component types``
  (the building blocks of a facility) and link it to existing or
  custom fragility algorithms.
- Component Criticality: Scenario Analysis tools allow users to
  identify the cost of restoration for chosen scenarios, expected
  restoration times, and which component upgrades can most benefit
  the system.
- Restoration Prognosis: User can experiment with different levels of
  hazards and post-disaster resource allocation to gauge restoration
  times for facility operations.

Contents:

.. toctree::
   :maxdepth: 2

   ch01_intro
   ch02_concept_design
   ch03_installation
   ch04_simsetup


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

