.. _intro-page:

************
Introduction
************

Critical infrastructure assets typically comprise a number of 
interconnected components that work in concert to deliver a service. 
In the context of natural hazard vulnerability, the components have differing 
susceptibilities, require different resource levels and time to repair, and 
have a range of criticalities to the overall service delivery. The 
vulnerability of the asset, then, is a product of their constituent components, 
their properties and interactions.

SIRA stands for 'Systemic Infrastructure Resilience Analysis'. 
It comprises a method and software tools that provide a framework for 
simulating the fragility of infrastructure assets to natural hazards, 
based on assessment of the fragilities and configuration of components that 
comprises the asset. To date system has been used to work with 
earthquake hazards only. SIRA enables the vulnerabilities of each element
to be within a facility or a network to be integrated into a holistic
assessment of the direct system losses, service disruption and cost.

SIRA is designed for simulating vulnerability of high-value 
infrastructure systems to natural hazards. The infrastructure types can be 
individual sites or facilities (e.g., an electrical substation, or a 
water treatment plant) or networks (e.g., an electricity transmission network, 
water transmission network. etc.). Earthquake ground motion has been the 
primary focus. Uncertainties are captured through a Monte Carlo sampling
process. The tool facilitates quantification of infrastructure assets' 
vulnerability, and also enables the most vulnerable components to be identified 
in terms of repair cost, time to recovery, and implications for service 
continuity. The tool has also been designed and tested to assess risk, but 
that functionality would typically require that the simulation be run in 
high performance computing environment.

The outcomes of this tool can support identification of retrofit options, 
and their relative efficacy in reducing risk. SIRA can be used in generating 
information for cost-benefit analyses of retrofit options, which can support 
asset managers in decisions around the most cost-effective utilisation of 
limited retrofit resources.


.. _intro-design-notes:

:term:`Vulnerability <vulnerability>` of a facility is modelled by assigning 
fragilities to the individual components that make up a facility or a network. 
The program accounts for variability in component fragilities by sampling 
probability distributions for the each fragility curve median and beta values. 
Once values have been selected for each curve it checks that 
:term:`fragility curves` do not overlap and if they do, re-samples the 
median and beta probability distributions until non-overlapping fragility 
curves are produced.

Damage scales for most facility types, along with the recovery time estimation 
method, has been taken from HAZUS :cite:`FEMA2003`. Although, where deemed more 
appropriate, custom damage scales have been used, e.g. for electrical 
substations. Repair cost (and hence damage index) and recovery times for each 
component are customised for each asset type, based on consultation with 
assets operators. The threshold values of spectral acceleration for each 
of four damage states are sampled by randomly sampling the fragility curves 
described above.

Please note that hazard modelling is done externally by relevant experts 
using other applications, and used in this tool as input.
