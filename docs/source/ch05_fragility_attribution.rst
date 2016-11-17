.. _fragility-attribution:

*******************************
Component Fragility Attribution
*******************************

As earthquake induced ground shaking at a facility increases in intensity,
the individual components respond and sustain progressively more damage.
Fragility functions are used to define this susceptibility of components
to damage by quantifying the likelihood that a level of damage will be
exceeded for a given level of shaking. This approach entails the definition
of one or more earthquake damage states for each component and the selection
of a ground shaking measure that is highly correlated to the component damage.

Each component within a facility is strictly unique. However, for a given
component it can be classed into a category of asset referred to as a
``component type``, in which all components have similar vulnerability.
The assignment of component type fragility algorithms for each damage state
is based on asset typology and the design levels under consideration.
The fragility algorithm defines a cumulative probability of exceedance for
each damage state that captures the variability of the vulnerability within
a “component type” and the effect of variability in component response to
ground motion on damage severity.  The form of these functions can take a
range of mathematical forms.  The most common form is log normal and defined
by a median (|theta|) and log standard deviation (|beta|), which is the form
that SIFRA uses.

Component type fragility models need to be representative of the assets they
characterise. Models can be developed or sourced using the following
hierarchy of approaches which is in order of increasing uncertainty:

1. Direct consultation with industry asset managers to reach agreement on
   component fragilities by adjusting the most appropriate published models
   and drawing upon construction specifications, manufacturer’s shake table
   proof testing (if available) and observed earthquake performance
   (if possible).

2. Selection of the most applicable model from published models in the
   literature.

3. Utilisation of heuristic engineering judgment in adapting damage models
   for other components assessed to have similar fragility. There are many
   fragility models published in the literature.  The literature review and
   model compilation produced by the Syner-G Project :cite:`Pitilakis-etal-2014`.
   sponsored by the European Commission presents a wide range of electricity
   sector component models.  Its key references included works by
   Vanzi :cite:`Vanzi1996`, Anagnos :cite:`Anagnos1999` and
   the HAZUS Technical Manual :cite:`FEMA2003`.
   HAZUS includes fragility functions for a wider range of facility
   components than those presented in the Syner-G Project.

Each damage state has a description of the typical severity of physical
component damage which has implications for the use of the component.
Hence, each damage state also has an operational level for the damaged
component, a cost to repair (as a proportion of the replacement cost) and
a resource requirement in terms of number of technicians and time. This
is used to assess the utility of the facility immediately after an extreme
event and to assess the restoration prognosis.
