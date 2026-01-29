.. SIRA documentation master file.
   You can adapt this file completely to your liking,
   but it should at least contain the root `toctree` directive.

######################################################################
SIRA: A Methodology for Lifeline Infrastructure Vulnerability Analysis
######################################################################

https://github.com/GeoscienceAustralia/sira |br|
Release: |release|

SIRA stands for **Systemic Infrastructure Resilience Analysis**.
It represents a methodology and supporting code for systematising
:term:`vulnerability` analysis of lifeline infrastructure assets to
natural hazards (i.e. infrastructure system response to environmental
excitation).

The impact assessment incorporates the :term:`fragilities <fragility>`
of individual system components and their topological configuration
within the infrastructure system under study. This analysis is supplemented
by an assessment of the post-hazard system functionality through
network flow analysis, and consequent restoration times and costs.

The primary focus has been on studying responses of infrastructure systems,
both facilities (e.g. power generation plants, high voltage substations)
and geospatially dispersed networks (electricity transmission networks), to earthquake hazards.
The methodology is implemented in an open-source Python package.

Limitations: At the time of writing this documentation, the software has
been tested to work with earthquake hazards only. However, the methodology
and class structure have been developed to make the modelling process
hazard agnostic. The hazard specification process and infrastructure models
are designed to allow for expansion to different hazards and arbitrary
infrastructure sectors/assets.

SIRA is being developed in `Geoscience Australia (GA)`_ in support of
the agency's strategic priority to contribute to enhancing the hazard
resilience of communities in Australia and its region.


Features
========

**Open Source:** Written in Python, avoids dependency on proprietary tools,
platform agnostic.

**Flexible Infrastructure Model:** The data model is based on graph theory.
All infrastructure systems are represented as networks. This allows an user
to develop arbitrarily complex custom facility models -- for a :term:`facility`
or a network -- for impact simulation.

**Scalability:** The code can detect and leverage the host system capabilities
to enable parallel processing capabilities on high performace computing platforms
for very large scale simulations (tens of millions of hazard scenarios and model
with thousands of components), while being able to run on midrange laptops for
smaller hazard sets.

**Extensible Component Library:** User can define new instances of
`component_type` (the building blocks of a facility or network) and link it
to existing or custom fragility algorithms.

**Component Criticality Analysis:** Scenario Analysis tools allow users to
identify the cost of restoration for chosen scenarios, expected restoration
times, and options for targeted component upgrades that yield greatest
improvements to system resilience.

**Restoration Prognosis:** Users can experiment with different levels of
hazard intensities and post-disaster resource allocation to gauge restoration
times for system operations.


.. _user-docs:

.. toctree::
   :numbered:
   :maxdepth: 2
   :caption: User Documentation

   ch01_intro
   ch02_concept
   ch03_simsetup
   ch04_installation
   ch05_fragility_attribution


.. _back-matter:

.. toctree::
   :maxdepth: 1
   :caption: Back Matter

   copyrightnotice.rst
   bibliography.rst
   reportglossary.rst
