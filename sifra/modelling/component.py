# these are required for defining the data model
from collections import OrderedDict

from modelling.structural import jsonify
from sifra.modelling.structural import (
    Element,
    Base)


class IODict(OrderedDict, Base):
    def __init__(self, *args, **kwargs):
        if '_saved_values' in kwargs:
            super(IODict, self).__init__(kwargs['_saved_values'])
        else:
            super(IODict, self).__init__(*args, **kwargs)

        super(IODict, self).__init__(*args, **kwargs)
        self.key_index = {i: k for i, k in enumerate(self.iterkeys())}

    def __setitem__(self, key, value):
        super(IODict, self).__setitem__(key, value)
        self.key_index = {i: k for i, k in enumerate(self.iterkeys())}

    def index(self, index):
        return super(IODict, self).__getitem__(self.key_index[index])

    def __jsonify__(self):
        """
        Validate this instance and transform it into an object suitable for
        JSON serialisation.
        """
        res = {
            'class': [type(self).__module__, type(self).__name__],
            '_saved_values': {
                jsonify(k): jsonify(v)
                for k, v in self.iteritems()}}
        return res


class Component(Base):
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
    link_capacity = Element('float', 'Link capacity',0.0)
    weight = Element('float', 'Weight',0.0)


