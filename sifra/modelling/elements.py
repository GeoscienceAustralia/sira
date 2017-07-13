import six
from future.utils import itervalues
# these are required for defining the data model
from sifra.modelling.structural import (
    Element,
    Info,
    Base)



class ResponseModel(Base):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('__call__ is not implemented on {}'.format(
            self.__class__.__name__))


class DamageState(ReponseModel):
    damage_state_description = Element('str', 'A description of what the damage state means.')


class DamageAlgorithm(Base):
    damage_states = Element('dict', 'Response models for the damage stages',
        [lambda x: [isinstance(y, DamageState) for y in itervalues(x)]])

    def __call__(self, intensity_param, state):
        return self.damage_states[state](intensity_param)


class Component(Base):
    frag_func = Element('DamageAlgorithm', 'Fragility algorithm', Element.NO_DEFAULT)
    recovery_func = Element('DamageAlgorithm', 'Recovery algorithm', Element.NO_DEFAULT)

    def expose_to(self, intensity_param):
        return self.frag_func(intensity_param)


class Model(Base):
    description = Info('Represents a model (e.g. a "model of a powerstation")')

    components = Element('dict', 'A component', dict,
        [lambda x: [isinstance(y, Component) for y in itervalues(x)]])

    name = Element('str', "The model's name", 'model')

    def add_component(self, name, component):
        self.components[name] = component


def expose_to(self, intensity_param):
    return [c.expose_to(intensity_param) for c in itervalues(components)]
