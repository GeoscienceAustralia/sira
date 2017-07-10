from sifra.modelling.elements import ResponseModel
from sifra.modelling.structures import XYPairs
from sifra.modelling.structural import Element as _Element


class StepFunc(ResponseModel):
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


class LogNormalCDF(ResponseModel):
    median = _Element('float', 'Median of the log normal CDF.',
            _Element.NO_DEFAULT, [lambda x: float(x) > 0.])

    beta = _Element('float', 'Log standard deviation of the log normal CDF',
            _Element.NO_DEFAULT, [lambda x: float(x) > 0.])

    def __call__(self, value):
        """
        In scipy lognormal CDF is implemented thus:
            scipy.stats.lognorm.cdf(x, s, loc=0, scale=1)
        where,
            s = sigma   # or beta or standard deviation
            scale = exp(mean) = median
            loc is used to shift the distribution and commonly not used
        """
        import scipy.stats as stats
        return stats.lognorm.cdf(value, self.beta, loc=0, scale=self.median)


class NormalCDF(ResponseModel):
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
        import scipy.stats as stats
        return stats.norm.cdf(value, loc=self.mean, scale=self.stddev)

