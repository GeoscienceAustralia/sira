from sifra.modelling.structural import Info, Base


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
