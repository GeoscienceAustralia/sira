# these are required for defining the data model
from sifra.modelling.structural import (
    Element,
    Base)

from sifra.modelling.iodict import IODict

from sifra.modelling.responsemodels import DamageAlgorithm, RecoveryAlgorithm


class ExposableInfrastructure(object):
    """
    In design stage, not currently implemented
    The intention is to implement the behavioural logic for infrastructure
    under this class
    """
    def __init__(self, component):
        self.component = component
        self.hazard_levels = None

    def expose_to(self, hazard_level, scenario):
        pass

    def get_damage_state(self, hazard_level):
        pass

    def get_economic_loss(self, hazard_level):
        pass

    def get_functionality(self, hazard_level):
        pass


class Component(Base):
    """Fundamental unit of an infrastructure system. It
    contains the categorical definitions of the component
    as well as the algorithms that assess the damage and recovery."""
    component_type = Element('str', 'Type of component.')
    component_class = Element('str', 'Class of component.')
    node_type = Element('str', 'Class of node.')
    node_cluster = Element('str', 'Node cluster.')
    operating_capacity = Element('str', 'Node cluster.')

    frag_func = Element('DamageAlgorithm', 'Fragility algorithm', Element.NO_DEFAULT)
    recovery_func = Element('RecoveryAlgorithm', 'Recovery algorithm', Element.NO_DEFAULT)

    destination_components = Element('IODict', 'List of connected components', IODict)

    def expose_to(self, hazard_level, scenario):
        component_pe_ds = self.frag_func.pe_ds(hazard_level)

        return component_pe_ds

    def get_damage_state(self, ds_index):
        return self.frag_func.damage_states.index(ds_index)

    def get_recovery(self, ds_index):
        return self.recovery_func.recovery_states.index(ds_index)


class ConnectionValues(Base):
    link_capacity = Element('float', 'Link capacity', 0.0)
    weight = Element('float', 'Weight', 0.0)


