import logging

import numpy as np

from sira.modelling.iodict import IODict
from sira.modelling.responsemodels import Algorithm
from sira.modelling.structural import Base, Element

rootLogger = logging.getLogger(__name__)


class DamageState(Base):
    damage_state_name = Element("str", "Name of the damage state.")
    functionality = Element(
        "float", "Level of functionality for this damage level", 1.0, [lambda x: float(x) >= 0.0]
    )
    damage_ratio = Element("float", "damage ratio", 0.0, [lambda x: float(x) >= 0.0])

    response_function_constructor = Element("json", "Definition of damage function.")
    recovery_function_constructor = Element("json", "Definition of recovery function.")

    response_function = Element("ResponseAlgorithm", "Algorithm.")
    recovery_function = Element("RecoveryAlgorithm", "Algorithm.")

    def __init__(self, config=None, **kwargs):
        super(DamageState, self).__init__()

        for k, v in kwargs.items():
            if k == "response_function_constructor":
                self.response_function = Algorithm.factory(v)

            if k == "recovery_function_constructor":
                self.recovery_function = Algorithm.factory(v)
                # self._build_discrete_recovery_function(v, config)

            setattr(self, k, v)


class Component(Base):
    """
    Fundamental unit of an infrastructure system. It
    contains the categorical definitions of the component
    as well as the algorithms that assess the damage and recovery.
    """

    component_id = Element("str", "Id of component.")
    component_class = Element("str", "Class of component.")
    component_type = Element("str", "Type of component.")
    cost_fraction = Element("float", "Cost as a fraction of total value of system")
    node_cluster = Element("str", "Node cluster.")
    node_type = Element("str", "Class of node.")
    operating_capacity = Element("float", "Component nominal operating capacity")

    pos_x = Element("float", "Component locational value on the x-axis / longitude")
    pos_y = Element("float", "Component locational value on the y-axis / latitude")
    site_id = Element("int", "An integer value mapping to a hazard site identifier")

    damages_states_constructor = {}
    # Element('json', 'Information about damage states.')

    destination_components = Element("IODict", "List of connected components", IODict)

    damage_states = None

    def __init__(self, config, **kwargs):
        super(Component, self).__init__()

        self.damage_states = {}
        self.destination_components = IODict()

        for k, v in kwargs.items():
            setattr(self, k, v)

        for k, v in self.damages_states_constructor.items():
            # json file only takes string as keys, convert the
            # index representing the damage state to int
            params = {}
            for key in v.keys():
                params[key] = v[key]
            # Pass config and component info to DamageState
            params["_component_id"] = self.component_id
            params["_damage_state_index"] = int(k)
            self.damage_states[int(k)] = DamageState(config=config, **params)

    def get_location(self):
        return self.pos_x, self.pos_y, self.site_id

    def pe_ds(self, hazard_intensity):
        """
        Calculate the probabilities that this component will
        exceed the range of damage states.
        """

        pe_ds = np.zeros(len(self.damage_states))  # type: ignore

        for damage_state_index in self.damage_states.keys():
            pe_ds[damage_state_index] = self.damage_states[damage_state_index].response_function(
                hazard_intensity
            )

        return pe_ds

    def get_damage_state(self, ds_index):
        """
        Return the required damage state.
        :param ds_index: The index of the damage state,
            as produced by `expose_to` method. Integer value.
        :return: The fragility function object
        """
        return self.damage_states[ds_index]

    def get_recovery(self, ds_index):
        """
        The fragility function and recovery functions are both indexed
        on the same damage state index
        :param ds_index: Index of the damage state.
        :return: recovery function object
        """
        return self.damage_states[ds_index].recovery_function

    def __str__(self):
        damage_states = [
            self.damage_states[damage_state].damage_state_name
            for damage_state in self.damage_states  # type: ignore
        ]
        dest_comp_list = [
            destination_components
            for destination_components in self.destination_components  # type: ignore
        ]
        return f"""
            component_id           : {str(self.component_id)} '\n'
            component_class        : {str(self.component_class)} '\n'
            component_type         : {str(self.component_type)} '\n'
            cost_fraction          : {str(self.cost_fraction)} '\n'
            node_cluster           : {str(self.node_cluster)} '\n'
            node_type              : {str(self.node_type)} '\n'
            operating_capacity     : {str(self.operating_capacity)} '\n'
            pos_x                  : {str(self.pos_x)} '\n'
            pos_y                  : {str(self.pos_y)} '\n'
            damage_states          : {str(damage_states)} '\n'
            destination_components : {str(dest_comp_list)}'\n'
        """


class ConnectionValues(Base):
    """
    Each connection between two components has a capacity and
    a weight.
    """

    link_capacity = Element("float", "Link capacity", 1.0)
    weight = Element("float", "Weight", 0.0)

    def __str__(self):
        cv = f"""
        weight:        {str(self.weight)}
        link_capacity: {str(self.link_capacity)}
        """
        return cv
