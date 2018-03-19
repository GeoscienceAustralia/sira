import numpy as np
import scipy.stats as stats
from structural import Info, Base, Element
from sifra.modelling.structural import Element as _Element


class AlgorithmFactory:

    @staticmethod
    def factory(fragility_data): #algo_name, response_params):
        if len(fragility_data["damage_functions"]) <= 1:

            response_params = {}
            for key in fragility_data["damage_functions"][0].keys():
                response_params[key] = fragility_data["damage_functions"][0][key]
            algo_name = fragility_data["damage_functions"][0]["damage_function_name"]

            if algo_name == "StepFunc":
                return StepFunc(**response_params)
            elif algo_name == "LogNormalCDF":
                return LogNormalCDF(**response_params)
            elif algo_name == "NormalCDF":
                return NormalCDF(**response_params)
            elif algo_name == "ConstantFunction":
                return ConstantFunction(**response_params)
            elif algo_name == "Level0Response":
                return Level0Response(**response_params)
            elif algo_name == "Level0Recovery":
                return Level0Recovery()
        else:
            return PiecewiseFunction(**fragility_data)
    # add picewise funtion
        raise ValueError("No response model "
                         "matches {}".format(algo_name))




class DamageState(Base):
    """
    The allocated damage state for a given component
    Holds reference to the algorithum as value
    """

    damage_state = Element('str', 'Damage state name')
    # damage_state_description = Element('str', 'A description of what the damage state means')
    mode = Element('int','The mode used to estimate the damage')
    functionality = Element('float', 'Level of functionality for this damage level',
                            0.0, [lambda x: float(x) >= 0.0])
    # fragility_source = Element('str', 'The source of the parameter values')
    damage_ratio = Element('float', 'damage ratio',
                           0.0, [lambda x: float(x) >= 0.0])


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

class PiecewiseFunction(DamageState):
    """
    first function will only have one value if anything less than that always use that function
    last function will also have one value if anything greater than use that function
    in-between function will always have two range values they will only be defined for those values

    input: hazard value
    output: probability
    """

    def __init__(self, *arg, **kwargs):

        self.funtions = []
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

        for damage_function in self.damage_functions:
            if damage_function["damage_function_name"] == "StepFunc":
                self.funtions.append(StepFunc(**damage_function))
            elif damage_function["damage_function_name"] == "LogNormalCDF":
                self.funtions.append(LogNormalCDF(**damage_function))
            elif damage_function["damage_function_name"] == "NormalCDF":
                self.funtions.append(NormalCDF(**damage_function))
            elif damage_function["damage_function_name"] == "ConstantFunction":
                self.funtions.append(ConstantFunction(**damage_function))
            elif damage_function["damage_function_name"] == "Level0Response":
                self.funtions.append(Level0Response(**damage_function))
            elif damage_function["damage_function_name"] == "Level0Recovery":
                self.funtions.append(Level0Recovery(**damage_function))


    def __call__(self, hazard_intensity):

        for i, function in enumerate(self.funtions):
            if i == 0:
                if hazard_intensity < function.lower_limit:
                    return self.funtions[0]
            elif i == len(self.funtions)-1:
                if hazard_intensity < function.upper_limit:
                    return self.funtions[-1](hazard_intensity)
            else:
                if hazard_intensity < function.upper_limit and hazard_intensity > function.lower_limit:
                    return self.funtions[i](hazard_intensity)


# class PWFunction(DamageState):
#     damage_functions = _Element('Funtions', 'A list of funtions.', list)
#
#     def __init__(self, functions):
#         """
#         :param functions: list of items with somehting like (bounds, function)
#
#         functions: list(((lower, upper), 'StepFunc', ), ...)
#         """
#
#         FUNC_MAP = {'StepFunc': StepFunc}
#         def get_damage_func(func_name):
#
#
#         for f1, f2 in zip(functions[:-1], functions[1:]):
#             if f1.bounds['upper_limit'] != f2.bounds['lower_limit']:
#                 raise Exception()
#
#         def __call__(self, intensity):
#             for f in self.damage_functions:
#                 if f.bounds['upper_limit'] < intensity:
#                     continue
#                 return f(intensity)
#
#     def __call__
