from sifra.modelling.utils import IODict
import numpy as np

# these are required for defining the data model
from sifra.modelling.structural import (
    Element,
    Info,
    Base)

from sifra.modelling.component import Component

class IFSystem(Base):
    name = Element('str', "The model's name", 'model')
    description = Info('Represents a model (e.g. a "model of a powerstation")')

    components = Element('IODict', 'The components that make up the infrastructure system', {},
        [lambda x: [isinstance(y, Component) for y in x.itervalues()]])

    def add_component(self, name, component):
        self.components[name] = component

    def expose_to(self, hazard_level, scenario):
        # index of damage state of components: from 0 to nds+1
        if scenario.run_context:  # test run
            prng = np.random.RandomState(int(hazard_level.hazard_intensity))
        else:
            prng = np.random.RandomState()

        num_elements = len(self.components)
        # index of the damage state!
        component_damage_state_ind = np.zeros((scenario.num_samples, num_elements), dtype=int)

        for index, component in enumerate(self.components.itervalues()):
            # get the probability of exceeding damage state for each component
            component_pe_ds = component.expose_to(hazard_level)
            rnd = prng.uniform(size=(scenario.num_samples, len(component_pe_ds)))
            component_damage_state_ind[:, index] = np.sum(component_pe_ds > rnd, axis=1)


        # iterate throught the samples
        for sample_number in range(scenario.num_samples):
            loss_list_all_comp = []
            cp_func = []
            component_ds = component_damage_state_ind[sample_number,:]
            for index, component in enumerate(self.components.itervalues()):
                # get the damage state for the component
                damage_state = component.get_damage_state(component_ds[index])
                loss = damage_state.damage_ratio * component.cost_fraction
                loss_list_all_comp.append(loss)
                cp_func.append(damage_state.functionality)

            economic_loss_array_single[i] = sum(loss_list_all_comp)

        return component_damage_state_ind


