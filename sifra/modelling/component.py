# these are required for defining the data model
from sifra.modelling.structural import (
    Element,
    Base)

from sifra.modelling.elements import (DamageAlgorithm, RecoveryAlgorithm)


class Component(Base):
    component_type = Element('str', 'Type of component.')
    component_class = Element('str', 'Class of component.')
    node_type = Element('str', 'Class of node.')
    node_cluster = Element('str', 'Node cluster.')
    operating_capacity = Element('str', 'Node cluster.')

    frag_func = Element('DamageAlgorithm', 'Fragility algorithm', Element.NO_DEFAULT)
    recovery_func = Element('RecoveryAlgorithm', 'Recovery algorithm', Element.NO_DEFAULT)

    destination_components = Element('dict', 'List of connected components', {})

    def expose_to(self, intensity_param):
        print("Exposed {} to {}".format(self.component_type, intensity_param))
        return "Good" # self.frag_func(intensity_param)


class ConnectionValues(Base):
    link_capacity = Element('float', 'Link capacity',0.0)
    weight = Element('float', 'Weight',0.0)