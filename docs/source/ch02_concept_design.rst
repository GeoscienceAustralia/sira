.. _concept-and-design:

******************
Concept and Design
******************

.. _facility-system-model:

Facility System Model
=====================

Facilities are modelled in the ``sifra`` methodology as a network of
components. This approach affords three key advantages:

(1) it allows for modelling the effect of impaired or destroyed components 
    on the operational capacity of the system, 
    
(2) it allows for using graph theory to assess the graduated capacity 
    degradation, and restoration, through modelling flow through the 
    network, and 
    
(3) it allows for detection of the most efficient 'paths', or sets of 
    components, through the network that need to be restored in order to 
    establish a link between input and output nodes, i.e. to restore the 
    flow through the network which represents the productive capacity of 
    the system.

The concept of components and facilities used in this exercise map closely 
to the typology of micro- and macro-components respectively, as defined in 
the Synerg-G program :cite:`SynerG2010,Pitilakis-etal-2014`, and align with
the definition of subsystems and systems as defined in
:cite:`Rinaldi-etal-2001`. Under this scheme, the components are
represented as nodes. Based on their role within the system, these nodes,
or components, are classified in four categories:

1. *Supply nodes*: These nodes represent the entry points into the system
   for required inputs or commodities. As for example, coal and water can
   be the required ‘commodities’ into a thermal power station. In the case
   of the substation, required input is electricity from power stations or
   other substations.

2. *Output nodes*: These nodes represent the exit points for the output of 
   the system. For example, in the case of the power station, the output
   nodes act as dummy loads - representing the energy consumers - connected
   to each of the step-up transformers. The sum of flow through the network
   measured at the output nodes represented the effective production
   (or operational) capacity of the facility.

3. *Dependency nodes*: These nodes represent the components that do not 
   directly participate in the production process of the system, or in
   the handling of system inputs, but are critical for system operations
   in some other capacity, e.g. system management or monitoring. The
   control building of a substation is an example of this.

4. *Transhipment nodes*: These are nodes that transform, transport, or
   store system inputs to give effect to processes that produces the outputs
   required of the system. Majority of the nodes within a system fall into 
   this category.

The component configuration and redundancies are captured as edges connecting
the nodes. Constraints on flow through specific paths, or sets of nodes, can 
be represented as capacities of edges connecting those nodes. Figure 2.1 
illustrates this concept for a thermal power station.

.. _pwrstn_schematic_diagram:

.. figure:: _static/images/pwrstn_schematic_diagram.png
   :alt: Power station schematic
   :align: center
   :width: 98%
   
   Schematic representation of a coal-fired power station

The 'edges', or inter-nodal connections, represent a link or a process for 
maintaining 'flow' of goods or services within the system, and thus their 
directionality is important. For the power station, the edges are 
unidirectional, since the inputs flow in one direction starting from the 
entry point into the system and are progressively transformed through the 
system to generate energy – the end product. However, a substation is an 
electrical network where electricity – the system 'commodity' – can flow in 
either direction through an edge (electrical conductor) as dictated by load 
demands and system constraints. Therefore, most of the edges in the 
substation are bidirectional, unless specifically constrained.

Connection paths and 'production capacities' along those paths within a
system are calculated as the maximum flow through those paths.
The `igraph <http://igraph.org/python/>`_ Python package was used as the 
network modelling platform to calculate graph metrics for a post-hazard 
damaged system model. 


System Loss Modelling
=====================

For a given value of level of ground shaking, a set of random samples is 
generated, and the damage state of each component is calculated for each 
random sample based on the fragility function of the given component.
Given the assessed damage state of all the system components, the system
functionality is assessed and system output level calculated. This process 
is run through a Monte Carlo process for the set of random samples to
assess the system response at the selected ground shaking intensity. To
obtain a characterisation of the system and develop fragility algorithms
for the system (e.g. the power station) the process is repeated for a
range of PGA values. This Process is shown in Figure 2.2.

.. _fig_hazard_loss_link:

.. figure:: _static/images/hazard_loss_link.png
   :alt: Linking hazard to damage and loss
   :align: center
   :width: 98%

   Schematic of process linking component damage assessment to
   loss projection

Four discrete sequential damage states are used for assessing system 
fragility, similar to those used in HAZUS (FEMA 2003): DS1 Slight, 
DS2 Moderate, DS3 Extensive, DS4 Complete. The damage scale used for a power 
station is based on ranges of economic loss as a percentage of total system 
value.

The probability of a component exceeding damage state :math:`ds` is calculated
using the log-normal cumulative distribution function (CDF) as shown in
equation below, for a PGA value of :math:`x` g:

.. math::

   P[D_s \mid PGA=x] = \Phi \left(\dfrac {ln(x) - \mu_{lnX}}{\sigma_{lnX}}\right)
                     = \Phi \left(\dfrac {ln(x) - \mu_{\theta}}{\beta}\right)

where, |theta| = median, and |beta| = logarithmic standard deviation.

For a component in damage state :math:`ds_i`, the corresponding loss is
calculated as:

.. math::

   L_{C, ds_i} = R_{C, ds_i} \times CF_C

where, |br|
:math:`R_{C, ds_i}` = `d` is the damage ratio for component `C`
at damage state :math:`ds_i`, and |br|
:math:`CF_C` = cost of component `C` as a proportional of total system cost.


System Restoration Model
========================

The restoration algorithms are defined as normal functions. An approximation 
of mean restoration time for each component at each damage level is 
attributed. The structural damage level definitions associated with the 
damage states are central to establishing a common understanding to 
facilitate the development of the restoration parameters.

The functionality :math:`F_C` of component C at t time units after impact
of an earthquake of PGA=x is calculated as a weighted combination of the
probability of the components being in each of the S sequential damage 
states used in the model and the estimated recovery at time t for the 
components based of the restoration model:

.. math:: F_{C|x} = \sum_{i=0}^{S} P[{ds}_i \mid PGA=x] \times R_i[t]

where, :math:`{i}` is the index of the damage state,
:math:`{\{i \in \mathbb{Z} \mid 0 \leq i \leq S\}}`.
The 'None' damage state is :math:`{i=0}`, and :math:`{i=S}` is the complete
or highest modelled damage state. :math:`R_i[t]` is the likely level of
restoration of functionality at time :math:`t` . Restoration level
:math:`R_i` can take on any value in the unit interval [0,1].

The simulation of the restoration prognosis is conducted based on a set of 
inputs and assumptions. The required data inputs to this process are:

- The system configuration
- The modelled scenario - seismic intensity value
- Impact simulation results - system component losses
- Restoration priority list - the order at which output lines should 
  be recovered

The process assumes that restoration is undertaken in stages, subject to 
the level of resources that can be made available and the order of repairs. 
In regard to this, the concept of '*Restoration Streams*' is used--the 
maximum number of components that can be worked on simultaneously. This is 
effectively a proxy representing the deployment of trained personnel and 
material for the repair tasks. Additional optional offsets can be factored 
in to capture specific contexts: 

a)  *Restoration Offset* – this is a time allowance for assessment of
    damage to the system and for securing the site to assure it is safe
    for commencement of repairs;

b)  *Testing and Commission Interval*: this is a time allowance for testing
    conformity with operational and safety parameters for the system, or a
    part thereof.

Given a set of restoration parameters and the restoration plan, the
consequent restoration time is calculated as follows:

1. Test if there is any available path between the set of required input 
   nodes (i.e. supply nodes) and the output node assigned the highest 
   priority to meet the demand at that node.
   
2. If no functional path is found, then identify the least expensive path(s) 
   that needs to be restored to meet demand at the output node. Within each 
   path, identify the functional status of the nodes (components), and 
   generate a repair list.
   
3. Iterate through the ordered output list, repeating steps 1 and 2 above. 
   Update the component repair list and produce a complete prioritised list 
   of components to repair or replace.
   
4. Simulate an ordered restoration process based on the above list and 
   user-specified resource constraints. If the process is using x resource 
   constraints, then whenever a component is restored (and the number of 
   unrepaired components is ≥x), the next component is added to the active 
   repair list, so that at any one time x repair tasks are in progress. This 
   process is repeated until all the paths are restored, i.e. until system 
   output capacity is restored to normal levels.

In order to restore full capacity at an output node, it may be necessary to 
restore more than one path, i.e. connect an output node to multiple input 
nodes. This can be understood through some simple examples. If the facility 
in question is a thermal power station, the functioning of the generator 
depends on both the supply of fuel (as the source of energy to be
transformed) and water (for cooling and for steam production to drive the
turbines). In case of a substation, a certain output node may have a demand 
of 300MW, but it might be that there are four incoming lines each bringing 
in bringing in 100MW of electricity from power plants. In this case, the 
designated output node must be linked to at least three of the input/supply 
nodes to meet its demand.

In addition to the core process of approximating restoration time, a 
routine for simulating component cannibalisation within a facility or
system has also been incorporated. Here we use cannibalisation to refer
to an exercise whereby an operator may move an undamaged component from
a low priority or redundant line to replace a damaged component on a
high priority line. This exercise may allow the operator to eliminate
the potentially long procurement or transportation time for a replacement
unit, and thereby expedite the restoration of the high priority lines.

The outputs from the restoration model are: 

a)  a simple Gantt chart with each component needing repair,

b)  restoration plot for each output line over time and the associated
    percentage of total system capacity rehabilitated, and

c)  total restoration time for each output line for a given restoration
    scheme.

