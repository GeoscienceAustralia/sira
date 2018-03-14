import numpy as np
import scipy.stats as stats
from structural import Info, Base, Element
from sifra.modelling.structural import Element as _Element


class Algorithm:

    def factory(algo_name, response_params):
        if algo_name == "Lognormal":
            return LogNormalCDF(**response_params)
        if algo_name == "Normal":
            return NormalCDF(**response_params)
        if algo_name == "StepFunc":
            return StepFunc(**response_params)
        if algo_name == "Level0Response":
            return Level0Response()
        raise ValueError("No response model "
                         "matches {}".format(algo_name))

    factory = staticmethod(factory)

class XYPairs(Base):
    """
    A list of float values that implement a step function.
    """
    description = Info("The (x, f(x)) pairs defining a step function.")

    def __init__(self, pairs):
        """
        Create the tuple list containing the float values.
        :param pairs: An iterable container of tuples containing floats
        """
        self.pairs = pairs

    def __iter__(self):
        """
        Return the XYPairs
        :return: iterator over the XYPairs
        """
        return iter(self.pairs)

    def __jsonify__(self):
        """Called by jsonify to serialise this data"""
        return {
            'class': [type(self).__module__, type(self).__name__],
            'pairs': [[float(p[0]), float(p[1])] for p in self.pairs]}

class ResponseModel(Base):
    """
    Statistical model that assesses the response (amount of likely
    damage) for a given hazard
    """
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('__call__ is not implemented'
                                  ' on {}'.format(self.__class__.__name__))


class DamageState(ResponseModel):
    """
    The allocated damage state for a given component
    """
    damage_state = Element('str', 'Damage state name')
    # damage_state_description = Element('str', 'A description of what the damage state means')
    mode = Element('int','The mode used to estimate the damage')
    functionality = Element('float', 'Level of functionality for this damage level',
                            0.0, [lambda x: float(x) >= 0.0])
    # fragility_source = Element('str', 'The source of the parameter values')
    damage_ratio = Element('float', 'damage ratio',
                           0.0, [lambda x: float(x) >= 0.0])


class StepFunc(DamageState):
    """
    A response model that does not have a cumulative distribution
    function, rather a series of steps for damage.
    """
    xys = _Element('XYPairs', 'A list of X, Y pairs.', list,
        [lambda xy: [(float(x), float(y)) for x, y in xy]])

    def __call__(self, value):
        """
        Note that intervals are closed on the right.
        """

        for x, y in self.xys:
            if value < x:
                return y

        raise ValueError('value is greater than all xs!')


class LogNormalCDF(DamageState):
    """
    The log normal CDF response model for components.
    """
    median = _Element('float', 'Median of the log normal CDF.',
            _Element.NO_DEFAULT, [lambda x: float(x) > 0.])

    beta = _Element('float', 'Log standard deviation of the log normal CDF',
            _Element.NO_DEFAULT, [lambda x: float(x) > 0.])

    def __call__(self, hazard_intensity):
        """
        In scipy lognormal CDF is implemented thus:
            scipy.stats.lognorm.cdf(x, s, loc=0, scale=1)
        where,
            s = sigma   # or beta or standard deviation
            scale = exp(mean) = median
            loc is used to shift the distribution and commonly not used
        """
        return stats.lognorm.cdf(hazard_intensity, self.beta, loc=0, scale=self.median)


class NormalCDF(DamageState):
    """
    The normal CDF response model for components
    """
    mean = _Element('float', 'Mean of the normal or Gaussian CDF',
                    _Element.NO_DEFAULT, [lambda x: float(x) >= 0.])

    stddev = _Element('float', 'Standard deviation of the normal CDF',
                      _Element.NO_DEFAULT, [lambda x: float(x) > 0.])

    def __call__(self, value):
        """
        In scipy normal CDF is implemented thus:
            scipy.stats.norm.cdf(x, loc=0, scale=1)
        where,
        loc = Mean
        scale = Standard Deviation i.e. square root of Variance
        """
        return stats.norm.cdf(value, loc=self.mean, scale=self.stddev)


class ConstantFunction(DamageState):
    """
    A function for defining a constant amplitude for a given range
    """
    amplitude = _Element('float', 'Constant amplitude of function',
                    _Element.NO_DEFAULT, [lambda x: float(x) >= 0.])

    def __call__(self, value):
        return np.ones_like(value) * self.amplitude


class PiecewiseDefinedFunction(DamageState):
    """
    x          : array of x values over which the function is calculated

    boundaries : a list of floats the define the edges of the ranges
                 within which each function is applicable

    funcnames  : a list of names of functions or distributions

    arglist    : a list of lists, each containing the parameters
                 that required by the corresponding functions

    These lists must meet the following criteria:
        len(funcnames) == len(arglist) == len(boundaries)-1

    """
    boundaries = []
    funcnames = []
    arglist = []

    def __call__(self, xvalues):
        indices = [np.amax(np.where(xvalues <= val)) for val in self.boundaries]
        funclist = []
        for i in range(len(indices) - 1):
            if i + 1 < len(indices) - 1:
                xtmp = xvalues[indices[i]:indices[i + 1]]
            else:
                xtmp = xvalues[indices[i]:]
            funclist.append(self.funcnames[i](xtmp, *self.arglist[i]))
        allfuncs = np.concatenate(funclist)
        return allfuncs


class Level0Response(DamageState):
    """
    Standard response for no damage.
    """
    mode = 1
    damage_ratio = 0.0
    functionality = 1.0
    beta = 0.0
    median = 1.0

    def __call__(self, hazard_level):
        return 0.0


class Level0Recovery(DamageState):
    """
    Standard recovery for no damage.
    """
    recovery_mean = 0.00001
    recovery_std = 0.00001

    def __call__(self, hazard_level):
        return 0.0


class DamageAlgorithm(Base):
    """
    The collection of damage states that will calculate
    the component's response to a hazard.
    """
    damage_states = Element('IODict', 'Response models for the damage states',
        [lambda x: [isinstance(y, DamageState) for y in x.itervalues()]])

    def pe_ds(self, hazard_intensity):
        """Calculate the probabilities that this component will
        exceed the range of damage states."""
        pe_ds = np.zeros(len(self.damage_states))

        for offset, damage_state in enumerate(self.damage_states.values()):
            if damage_state.mode != 1:
                raise RuntimeError("Mode {} not implemented".format(damage_state.mode))
            pe_ds[offset] = damage_state(hazard_intensity)
        return pe_ds


class RecoveryState(Base):
    """
    The recovery parameters for a given damage level.
    """
    recovery_mean = Element('float', 'Recovery mean',
                            0.0, [lambda x: float(x) > 0.0])
    recovery_std = Element('float', 'Recovery standard deviation',
                           0.0, [lambda x: float(x) > 0.0])
    # recovery_95percentile = Element('float', 'Recovery 95th percentile',
    #                                 0.0, [lambda x: float(x) > 0.0])


# TODO Complete the recovery algorithm
class RecoveryAlgorithm(Base):
    """
    Collection of recovery states for a component.
    """
    recovery_states = Element('IODict', 'Recovery models for the damage states',
        [lambda x: [isinstance(y, RecoveryState) for y in x.itervalues()]])

    def __call__(self, intensity_param, state):
        for recovery_state in self.recovery_states:
            recovery_state(intensity_param)

        return 1.0


# entry point
class AlgorithmFactory(object):
    def __init__(self):
        self.response_algorithms = dict()
        self.recovery_algorithms = dict()

    def get_response_algorithm(self, component_type, hazard_type):
        return self.response_algorithms[hazard_type][component_type]

    def add_response_algorithm(self, component_type, hazard_type, algorithm):
        if hazard_type not in self.response_algorithms:
            self.response_algorithms[hazard_type] = {}

        self.response_algorithms[hazard_type][component_type] = algorithm

    # def get_recovery_algorithm(self, component_type, hazard_type):
    #     return self.recovery_algorithms[hazard_type][component_type]

    def add_recovery_algorithm(self, component_type, hazard_type, algorithm):
        if hazard_type not in self.recovery_algorithms:
            self.recovery_algorithms[hazard_type] = {}

        self.recovery_algorithms[hazard_type][component_type] = algorithm
