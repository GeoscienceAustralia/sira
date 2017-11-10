# these are required for defining the data model
from sifra.modelling.structural import (
    Element,
    Info,
    Base)


class ResponseModel(Base):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('__call__ is not implemented on {}'.format(
            self.__class__.__name__))


class DamageState(ResponseModel):
    damage_state_description = Element('str', 'A description of what the damage state means.')


class DamageAlgorithm(Base):
    damage_states = Element('dict', 'Response models for the damage stages',
        [lambda x: [isinstance(y, DamageState) for y in x.itervalues()]])

    def __call__(self, intensity_param, state):
        return self.damage_states[state](intensity_param)
