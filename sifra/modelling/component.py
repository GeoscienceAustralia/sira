# these are required for defining the data model
from sifra.modelling.structural import (Element,Base)
from sifra.modelling.iodict import IODict
import numpy as np
from sifra.modelling.responsemodels import Algorithm


class DamageState(Base):
    damage_state_name = Element('str', 'Name of the damage state.')
    functionality = Element('float', 'Level of functionality for this damage level', 0.0, [lambda x: float(x) >= 0.0])
    damage_ratio = Element('float', 'damage ratio', 0.0, [lambda x: float(x) >= 0.0])

    response_function_constructor = Element('json', 'Definition of damage function.')
    recovery_function_constructor = Element('json', 'Definition of recovery function.')

    response_function = Element('ResponseAlgorithm', 'Algorithm.')
    recovery_function = Element('RecoveryAlgorithm', 'Algorithm.')

    def __init__(self, *arg, **kwargs):

        self.response_function = None
        self.recovery_function = None

        for k, v in kwargs.iteritems():
            if k == 'response_function_constructor':
                self.response_function = Algorithm.factory(v)
                print(self.response_function)
            if k == 'recovery_function_constructor':
                self.recovery_function = Algorithm.factory(v)
            setattr(self, k, v)
        print(self.response_function)


class Component(Base):
    """Fundamental unit of an infrastructure system. It
    contains the categorical definitions of the component
    as well as the algorithms that assess the damage and recovery."""

    component_id = Element('str', 'Id of component.')
    component_class = Element('str', 'Class of component.')
    component_type = Element('str', 'Type of component.')
    cost_fraction = Element('float', 'Cost as a fraction of total value of system')
    node_cluster = Element('str', 'Node cluster.')
    node_type = Element('str', 'Class of node.')
    operating_capacity = Element('float', 'Component nominal operating capacity')
    longitude = Element('float', 'Component longitude location value')
    latitude = Element('float', 'Component latitude location value')

    damages_states_constructor = {}  # Element('json', 'Information about damage states.')

    destination_components = Element('IODict', 'List of connected components', IODict)

    damage_states = None

    def __init__(self, *arg, **kwargs):

            self.damage_states = {}
            self.destination_components = IODict()

            for k, v in kwargs.iteritems():
                setattr(self, k, v)

            # TODO a check about the order of the damage state
            # the key of the damage state

            for k, v in self.damages_states_constructor.iteritems():
                # json file only takes string as keys, convert the index representing the damage state to int

                params = {}
                for key in v.keys():
                    params[key] = v[key]

                self.damage_states[int(k)] = DamageState(**params)

    def pe_ds(self, hazard_intensity):
        """Calculate the probabilities that this component will
        exceed the range of damage states."""

        pe_ds = np.zeros(len(self.damage_states))

        for damage_state_index in self.damage_states.keys():
            pe_ds[damage_state_index] = self.damage_states[damage_state_index].response_function(hazard_intensity)

        return pe_ds

    def get_damage_state(self, ds_index):
        """
        Return the required damage state.
        :param ds_index: The index of the damage state, as supplied by expose_to method. Integer value.
        :return: The fragility function object
        """
        return self.damage_states[ds_index]

    def get_recovery(self, ds_index):
        """
        The fragility function and recovery functions are both indexed on the same damage
        state index
        :param ds_index: Index of the damage state.
        :return: recovery function object
        """
        return self.damage_states[ds_index].recovery_function


class ConnectionValues(Base):
    """
    Each connection between two components has a capacity and
    a weight.
    """
    # link_capacity = Element('float', 'Link capacity', 0.0)
    # weight = Element('float', 'Weight', 0.0)
