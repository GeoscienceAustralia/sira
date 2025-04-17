import numpy as np
import scipy.stats as stats
from sira.modelling.structural import Base
from sira.modelling.structural import Element as _Element
from sira.modelling.structural import Info


class Algorithm:

    @staticmethod
    def factory(response_params):

        function_name = response_params["function_name"]
        funcname_nocase = str(function_name).casefold()

        if funcname_nocase in [
                "stepfunc", "step_func", "stepfunction", "step_function"]:
            return StepFunc(**response_params)

        elif funcname_nocase in [
                "lognormal", "lognormalcdf", "lognormal_cdf"]:
            return LogNormalCDF(**response_params)

        elif funcname_nocase in [
                "normal", "normalcdf", "normal_cdf"]:
            return NormalCDF(**response_params)

        elif funcname_nocase in [
                "rayleigh", "rayleighcdf", "rayleigh_cdf"]:
            return RayleighCDF(**response_params)

        elif funcname_nocase in [
                "ConstantFunction".lower(), "constant_function"]:
            return ConstantFunction(**response_params)

        elif funcname_nocase in [
                "Level0Response".lower(), "Level0Recovery".lower()]:
            return Level0Response(**response_params)

        elif funcname_nocase in [
                "PiecewiseFunction".lower(), "piecewise_function"]:
            return PiecewiseFunction(**response_params)

        raise ValueError("No response model matches {}".format(function_name))


class Level0Response(Base):
    """
    Standard response for no damage.
    """
    mode = 1
    damage_ratio = 0.0
    functionality = 1.0
    beta = 0.0
    median = 1.0

    lower_limit = _Element(
        'float',
        'lower limit of function if part of piecewise function',
        None, [lambda x: float(x) > 0.])
    upper_limit = _Element(
        'float',
        'upper limit of function  if part of piecewise function',
        None, [lambda x: float(x) > 0.])

    def __call__(self, hazard_level):
        return 0.0


class RayleighCDF(Base):
    """
    The Rayliegh CDF response model for components.
    """
    scale = _Element(
        'float',
        'Scale parameter for Rayleigh CDF.',
        _Element.NO_DEFAULT, validators=[lambda x: float(x) > 0.])

    loc = _Element(
        'float',
        'Location parameter for Rayleigh CDF.',
        default=0, validators=[lambda x: float(x) >= 0.])

    def __call__(self, x):
        """
        SciPy implementation of Rayleigh CDF:
        loc = shift parameter
        scale = scaling parameter
        """
        return stats.rayleigh.cdf(x, loc=self.loc, scale=self.scale)


class LogNormalCDF(Base):
    """
    The lognormal CDF response model for components.
    """

    median = _Element(
        'float', 'Median of the log normal CDF.',
        _Element.NO_DEFAULT, [lambda x: float(x) > 0.])

    beta = _Element(
        'float', 'Log standard deviation of the log normal CDF',
        _Element.NO_DEFAULT, [lambda x: float(x) > 0.])

    location = _Element(
        'float', 'Location parameter of the log normal CDF',
        0.0, [lambda x: float(x) > 0.])

    lower_limit = _Element(
        'float',
        'lower limit of function if part of piecewise function',
        None,
        [lambda x: float(x) > 0.])

    upper_limit = _Element(
        'float',
        'upper limit of function  if part of piecewise function',
        None,
        [lambda x: float(x) > 0.])

    def __call__(self, data_point):
        """
        SciPy implementation of LogNormal CDF:
            scipy.stats.lognorm.cdf(x, s, loc=0, scale=1)
        where,
            s = sigma   # or beta or standard deviation (shape parameter)
            scale = exp(mean) = median
            loc is used to shift the distribution and commonly not used
        """
        return stats.lognorm.cdf(
            data_point, self.beta, loc=self.location, scale=self.median)


class NormalCDF(Base):
    """
    The normal CDF response model for components
    """
    # -----------------------------------------------
    mean = _Element(
        'float',
        'Mean of the normal or Gaussian CDF',
        _Element.NO_DEFAULT,
        [lambda x: float(x) >= 0.])
    stddev = _Element(
        'float',
        'Standard deviation of the normal CDF',
        _Element.NO_DEFAULT,
        [lambda x: float(x) > 0.])
    # -----------------------------------------------
    lower_limit = _Element(
        'float',
        'lower limit of function if part of piecewise function',
        -np.inf,
        [lambda x: float(x) > 0.])
    upper_limit = _Element(
        'float',
        'upper limit of function  if part of piecewise function',
        np.inf,
        [lambda x: float(x) > 0.])
    # -----------------------------------------------

    def __call__(self, data_point, inverse=False):
        """
        SciPy implementation of Normal CDF:
            scipy.stats.norm.cdf(x, loc=0, scale=1)
        where,
        loc = Mean
        scale = Standard Deviation i.e. square root of Variance
        """
        if not inverse:
            return stats.norm.cdf(data_point, loc=self.mean, scale=self.stddev)
        elif inverse:
            return stats.norm.ppf(data_point, loc=self.mean, scale=self.stddev)


class ConstantFunction(Base):
    """
    A function for defining a constant amplitude for a given range
    """
    amplitude = _Element(
        'float',
        'Constant amplitude of function',
        _Element.NO_DEFAULT, [lambda x: float(x) >= 0.])

    lower_limit = _Element(
        'float',
        'lower limit of function if part of piecewise function',
        None, [lambda x: float(x) >= 0.])
    upper_limit = _Element(
        'float',
        'upper limit of function  if part of piecewise function',
        None, [lambda x: float(x) >= 0])

    def __call__(self, hazard_intensity):
        return self.amplitude


class StepFunc(Base):
    """
    A response model that does not have a cumulative distribution
    function, rather a series of steps for damage.
    """
    xys = _Element(
        'XYPairs', 'A list of X, Y pairs.', list,
        [lambda xy: [(float(x), float(y)) for x, y in xy]])

    lower_limit = _Element(
        'float',
        'lower limit of function if part of piecewise function',
        None,
        [lambda x: float(x) > 0.])

    upper_limit = _Element(
        'float',
        'upper limit of function  if part of piecewise function',
        None,
        [lambda x: float(x) > 0.])

    def __call__(self, hazard_intensity):
        """
        Note that intervals are closed on the right.
        """
        for x, y in self.xys:  # noqa: E1133
            if hazard_intensity < x:
                return y
        raise ValueError('value is greater than all xs!')


class XYPairs(object):
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


class PiecewiseFunction(object):
    """
    This class builds a piecwise function defined by algorithm constructor
    data of a specified format. This data is part of the defined
    attributes of a system Component.

    Each dict in the list contains:
        - the parameters required to construct an algorithm, and
        - the conditions where that algorithm will be applicable
    """
    piecewise_function_constructor = None

    def __init__(self, **kwargs):
        """
        input: a list of dicts.
        Dict name must be 'piecewise_function_constructor'
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.functions = []
        self.validranges = []
        for param_dict in self.piecewise_function_constructor:  # noqa: E1133
            lo = self.check_limit(param_dict['lower_limit'], which_lim='lower')
            hi = self.check_limit(param_dict['upper_limit'], which_lim='upper')
            self.functions.append(Algorithm.factory(param_dict))
            self.validranges.append((lo, hi))

    def check_limit(self, val, which_lim):
        if which_lim == 'lower':
            inf, infstr = -np.inf, ['-np.inf', '-inf']
        else:
            inf, infstr = np.inf, ['np.inf', '+np.inf', 'inf', '+inf']

        if (val is None) or str(val) in ['', 'NA', *infstr]:
            val = inf
        else:
            try:
                val = float(val)
            except ValueError:
                print(f"Invalid value passed for {which_lim} limit of function.")
                exit(1)
        return val

    def condfunc(self, x, func_lims):
        return (x >= func_lims[0]) & (x < func_lims[1])

    def pwfunc(self, x):
        x = np.asarray(x)
        y = np.zeros(x.shape)
        for i, func in enumerate(self.functions):
            func_lims = self.validranges[i]
            y += self.condfunc(x, func_lims) * func(x)   # noqa: W0123
        return y

    def __call__(self, hazard_intensity):
        """
        input: hazard intensity value
        output: probability of a response (linked to a damage state)
        """
        vectorized_pwf = np.vectorize(self.pwfunc)
        return vectorized_pwf(hazard_intensity)
