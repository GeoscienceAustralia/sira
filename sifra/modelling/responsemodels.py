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
        import scipy.stats as stats
        return stats.lognorm.cdf(value, self.beta, scale=self.median)
