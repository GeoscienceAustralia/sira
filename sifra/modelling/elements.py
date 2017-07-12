import numpy as np
from sifra.modelling.utils import IODict

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
    damage_state = Element('str', 'Damage state name')
    damage_state_description = Element('str', 'A description of what the damage state means')
    mode = Element('int','The mode used to estimate the damage')
    functionality = Element('float', 'Level of functionality for this damage level',
                            0.0, [lambda x: float(x) >= 0.0])
    fragility_source = Element('str', 'The source of the parameter values')
    damage_ratio = Element('float', 'damage ratio',
                            0.0, [lambda x: float(x) >= 0.0])


class DamageAlgorithm(Base):
    damage_states = Element('IODict', 'Response models for the damage states',
        [lambda x: [isinstance(y, DamageState) for y in x.itervalues()]])

    def pe_ds(self, intensity_param):
        pe_ds = np.zeros(len(self.damage_states))

        for offset, damage_state in enumerate(self.damage_states.itervalues()):
            if damage_state.mode != 1:
                raise RuntimeError("Mode {} not implemented".format(damage_state.mode))
            pe_ds[offset] = damage_state(intensity_param)

        return pe_ds


class RecoveryState(Base):
    recovery_mean = Element('float', 'Recovery mean',
                            0.0, [lambda x: float(x) > 0.0])
    recovery_std = Element('float', 'Recovery standard deviation',
                            0.0, [lambda x: float(x) > 0.0])
    recovery_95percentile = Element('float', 'Recovery 95th percentile',
                                    0.0, [lambda x: float(x) > 0.0])


class RecoveryAlgorithm(Base):
    recovery_states = Element('IODict', 'Recovery models for the damage states',
        [lambda x: [isinstance(y, RecoveryState) for y in x.itervalues()]])

    def __call__(self, intensity_param, state):
        for recovery_state in self.recovery_states:
            recovery_state(intensity_param)

        return
