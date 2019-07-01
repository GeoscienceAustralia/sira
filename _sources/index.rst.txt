.. SIFRA documentation master file, created by
   sphinx-quickstart on Thu Feb 25 09:28:33 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#######################################################################
SIFRA: A Methodology for Lifeline Infrastructure Vulnerability Analysis
#######################################################################

https://github.com/GeoscienceAustralia/sifra |br|
Release: |release|

SIFRA stands for **System for Infrastructure Facility Resilience Analysis**.
It represents a methodology and supporting code for systematising
:term:`vulnerability` analysis of lifeline infrastructure to natural hazards
(i.e. response of infrastructure assets to environmental excitation).

The impact assessment is based on the :term:`fragilities <fragility>`
and configuration of components that comprise the infrastructure system
under study. The analytical process is supplemented by an assessment of
the system functionality through the post-damage network flow analysis,
and approximations for recovery timeframes.

The current focus has been on studying responses of infrastructure facilities
(e.g. power generation plants, high voltage substations). Considerable
work has been done in the code backend to extend the same methodology
to modelling network vulnerability as well
(e.g. electricity transmission networks).

Currently the software has been tested to work with earthquake hazards only.
However, the methodology and class structure have been developed such
that the platform is effectively hazard agnostic. The hazard attribution
process and infrastructure models are flexible to allow for expansion
to other hazards and new infrastructure sectors.

SIFRA was developed in `Geoscience Australia (GA) <http://www.ga.gov.au/>`_
in support of the agency's vision to contribute to enhancing the resilience
of communities in Australia and its region.


Features
========

*Open Source:*
    Written in Python, and there is no dependency on
    proprietary tools. It should run on OS X, Windows, and
    Linux platforms.

*Flexible Infrastructure Model:*
    The data model is based on graph theory. All infrastructure systems
    are represented as networks. This allows an user to develop
    arbitrarily complex custom facility models -- for a :term:`Facility`
    or a network -- for impact simulation.

*Extensible Component Library:*
    User can define new instances of `component_type`
    (the building blocks of a facility) and link it to existing or
    custom fragility algorithms.

*Component Criticality Analysis:*
    Scenario Analysis tools allow users to identify the cost of
    restoration for chosen scenarios, expected restoration times,
    and which component upgrades can most benefit the system.

*Restoration Prognosis:*
    User can experiment with different levels of hazards and
    post-disaster resource allocation to gauge restoration
    times for facility operations.


.. _user-docs:

.. toctree::
   :numbered:
   :maxdepth: 2
   :caption: User Documentation

   ch01_intro
   ch02_concept
   ch03_installation
   ch04_simsetup
   ch05_fragility_attribution


.. _back-matter:

.. toctree::
   :maxdepth: 1
   :caption: Back Matter

   copyrightnotice.rst
   bibliography.rst
   reportglossary.rst
