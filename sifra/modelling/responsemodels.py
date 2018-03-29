import scipy.stats as stats
from structural import Info, Base, Element
from sifra.modelling.structural import Element as _Element


class Algorithm:

    @staticmethod
    def factory(response_params):

        function_name = response_params["function_name"]
        if function_name == "StepFunc":
            return StepFunc(**response_params)
        elif function_name == "LogNormalCDF" or function_name == "Lognormal":
            return LogNormalCDF(**response_params)
        elif function_name == "NormalCDF":
            return NormalCDF(**response_params)
        elif function_name == "ConstantFunction":
            return ConstantFunction(**response_params)
        elif function_name == "Level0Response":
            return Level0Response(**response_params)
        elif function_name == "Level0Recovery":
            return Level0Recovery()
        elif function_name == "PiecewiseFunction":
            return PiecewiseFunction(**response_params)
        elif function_name == "RecoveryFunction":
            return RecoveryFunction(**response_params)

        raise ValueError("No response model matches {}".format(function_name))


class RecoveryFunction(Base):

    recovery_mean = Element('float', 'Recovery mean',
                            0.0, [lambda x: float(x) > 0.0])
    recovery_std = Element('float', 'Recovery standard deviation',
                           0.0, [lambda x: float(x) > 0.0])
    recovery_95percentile = Element('float', 'Recovery 95th percentile',
                                    0.0, [lambda x: float(x) > 0.0])

    def __call__(self, intensity_param, state):
        return 1.0


class StepFunc(Base):
    """
    A response model that does not have a cumulative distribution
    function, rather a series of steps for damage.
    """
    xys = _Element('XYPairs', 'A list of X, Y pairs.', list,
        [lambda xy: [(float(x), float(y)) for x, y in xy]])

    lower_limit = _Element('float', 'lower limit of function if part of piecewise function',
                           None, [lambda x: float(x) > 0.])
    upper_limit = _Element('float', 'upper limit of function  if part of piecewise function',
                           None, [lambda x: float(x) > 0.])

    def __call__(self, hazard_intensity):
        """
        Note that intervals are closed on the right.
        """
        for x, y in self.xys:
            if hazard_intensity < x:
                return y
        raise ValueError('value is greater than all xs!')


class LogNormalCDF(Base):
    """
    The log normal CDF response model for components.
    """

    lower_limit = _Element('float', 'lower limit of function if part of piecewise function',
                           None, [lambda x: float(x) > 0.])
    upper_limit = _Element('float', 'upper limit of function  if part of piecewise function',
                           None, [lambda x: float(x) > 0.])

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


class NormalCDF(Base):
    """
    The normal CDF response model for components
    """
    mean = _Element('float', 'Mean of the normal or Gaussian CDF',
                    _Element.NO_DEFAULT, [lambda x: float(x) >= 0.])

    stddev = _Element('float', 'Standard deviation of the normal CDF',
                      _Element.NO_DEFAULT, [lambda x: float(x) > 0.])

    lower_limit = _Element('float', 'lower limit of function if part of piecewise function',
                           None, [lambda x: float(x) > 0.])
    upper_limit = _Element('float', 'upper limit of function  if part of piecewise function',
                           None, [lambda x: float(x) > 0.])

    def __call__(self, hazard_intensity):
        """
        In scipy normal CDF is implemented thus:
            scipy.stats.norm.cdf(x, loc=0, scale=1)
        where,
        loc = Mean
        scale = Standard Deviation i.e. square root of Variance
        """
        return stats.norm.cdf(hazard_intensity, loc=self.mean, scale=self.stddev)


class ConstantFunction(Base):
    """
    A function for defining a constant amplitude for a given range
    """
    amplitude = _Element('float', 'Constant amplitude of function',
                    _Element.NO_DEFAULT, [lambda x: float(x) >= 0.])

    lower_limit = _Element('float', 'lower limit of function if part of piecewise function',
                           None, [lambda x: float(x) > 0.])
    upper_limit = _Element('float', 'upper limit of function  if part of piecewise function',
                           None, [lambda x: float(x) > 0.])

    def __call__(self, hazard_intensity):
        return self.amplitude


class Level0Response(Base):
    """
    Standard response for no damage.
    """
    mode = 1
    damage_ratio = 0.0
    functionality = 1.0
    beta = 0.0
    median = 1.0

    lower_limit = _Element('float', 'lower limit of function if part of piecewise function',
                           None, [lambda x: float(x) > 0.])
    upper_limit = _Element('float', 'upper limit of function  if part of piecewise function',
                           None, [lambda x: float(x) > 0.])

    def __call__(self, hazard_level):
        return 0.0


class Level0Recovery(Base):
    """
    Standard recovery for no damage.
    """
    recovery_mean = 0.00001
    recovery_std = 0.00001

    lower_limit = _Element('float', 'lower limit of function if part of piecewise function',
                           None, [lambda x: float(x) > 0.])
    upper_limit = _Element('float', 'upper limit of function  if part of piecewise function',
                           None, [lambda x: float(x) > 0.])

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


class PiecewiseFunction(Base):
    """
    first function will only have one value if anything less than that always use that function
    last function will also have one value if anything greater than use that function
    in-between function will always have two range values they will only be defined for those values

    input: hazard value
    output: probability
    """
    piecewise_function_constructor = None
    piecewise_functions = None

    def __init__(self, *arg, **kwargs):

        self.piecewise_functions = []
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

        for function_constructor in self.piecewise_function_constructor:
            function_params = {}
            for key in function_constructor.keys():
                function_params[key] = function_constructor[key]

            self.piecewise_functions.append(Algorithm.factory(function_params))

    def __call__(self, hazard_intensity):

        for i, piecewise_function in enumerate(self.piecewise_functions):
            # check if lower limit function
            if i == 0:
                if hazard_intensity <= piecewise_function.lower_limit:
                    return self.piecewise_functions[0]
            # check if upper limit function
            elif i == len(self.piecewise_functions)-1:
                if hazard_intensity < piecewise_function.upper_limit:

                    return self.piecewise_functions[-1](hazard_intensity)
            # any other function between the limits
            else:
                if piecewise_function.lower_limit <= hazard_intensity < piecewise_function.upper_limit:
                    return self.piecewise_functions[i](hazard_intensity)