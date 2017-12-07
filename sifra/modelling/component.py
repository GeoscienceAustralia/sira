import numpy as np

# these are required for defining the data model
from sifra.modelling.structural import (Element,
                                        Base)

from sifra.modelling.iodict import IODict

from sifra.modelling.responsemodels import DamageAlgorithm, RecoveryAlgorithm
from sifra.modelling.location import Location


class Component(Base):
    """Fundamental unit of an infrastructure system. It
    contains the categorical definitions of the component
    as well as the algorithms that assess the damage and recovery."""

    component_type = Element('str', 'Type of component.')
    component_class = Element('str', 'Class of component.')
    node_type = Element('str', 'Class of node.')
    node_cluster = Element('str', 'Node cluster.')

    operating_capacity = Element('float', 'Component nominal operating capacity')
    cost_fraction = Element('float', 'Cost as a fraction of total value of system')

    frag_func = Element('DamageAlgorithm', 'Fragility algorithm', Element.NO_DEFAULT)
    recovery_func = Element('RecoveryAlgorithm', 'Recovery algorithm', Element.NO_DEFAULT)

    destination_components = Element('IODict', 'List of connected components', IODict)

    def expose_to(self, hazard_level, scenario):
        """
        Expose the component to the hazard using the
        damage algorithm.
        :param hazard_level: the hazard level
        :param scenario: Additional parameters that may be required to assess hazard damage.
        :return: The array of probabilities that each damage level was exceeded.
        """
        hazard_intensity = hazard_level.determine_intensity_at(self.get_location())

        component_pe_ds = self.frag_func.pe_ds(hazard_intensity)

        return component_pe_ds

    def get_damage_state(self, ds_index):
        """
        Return the required damage state.
        :param ds_index: The index of the damage state, as supplied by expose_to method.
        :return: The fragility function object
        """
        return self.frag_func.damage_states.index(ds_index)

    def get_recovery(self, ds_index):
        """
        The fragility function and recovery functions are both indexed on the same damage
        state index
        :param ds_index: Index of the damage state.
        :return: recovery function object
        """
        return self.recovery_func.recovery_states.index(ds_index)

    @staticmethod
    def get_location():
        return Location(np.NAN, np.NAN, np.NAN)


class ConnectionValues(Base):
    """
    Each connection between two components has a capacity and
    a weight.
    """
    link_capacity = Element('float', 'Link capacity', 0.0)
    weight = Element('float', 'Weight', 0.0)
