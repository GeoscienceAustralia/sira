import six
# from future.utils import itervalues
# these are required for defining the data model
from sifra.modelling.structural import (
    Element,
    Info,
    Base)


class ResponseModel(Base):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('__call__ is not implemented'
                                  ' on {}'.format(self.__class__.__name__))


class DamageState(ResponseModel):
    damage_state = Element('str', 'A description of what the damage state means.')
    damage_state_description = Element('str', 'A description of what the damage state means.')
    mode = Element('int','The mode used to estimate the damage')
    functionality = Element('float', 'Level of functionality for this damage level',
                            0.0, [lambda x: float(x) >= 0.0])


class DamageAlgorithm(Base):
    damage_states = Element('dict', 'Response models for the damage states',
        [lambda x: [isinstance(y, DamageState) for y in x.itervalues()]])

    def __call__(self, intensity_param, state):
        return self.damage_states[state](intensity_param)



class RecoveryState(Base):
    recovery_mean = Element('float', 'Recovery mean',
                            0.0, [lambda x: float(x) > 0.0])
    recovery_std = Element('float', 'Recovery standard deviation',
                            0.0, [lambda x: float(x) > 0.0])
    recovery_95percentile = Element('float', 'Recovery 95th percentile',
                                    0.0, [lambda x: float(x) > 0.0])


class RecoveryAlgorithm(Base):
    recovery_states = Element('dict', 'Recovery models for the damage states',
        [lambda x: [isinstance(y, RecoveryState) for y in x.itervalues()]])

    def __call__(self, intensity_param, state):
        return self.damage_states[state](intensity_param)
