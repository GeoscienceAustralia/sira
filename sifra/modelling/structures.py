from sifra.modelling.structural import Info, Base


class XYPairs(Base):
    description = Info("The (x, f(x)) pairs defining a step function.")

    def __init__(self, pairs):
        self.pairs = pairs

    def __iter__(self):
        return iter(self.pairs)

    def __jsonify__(self):
        return {
            'class': [type(self).__module__, type(self).__name__],
            'pairs': [[float(p[0]), float(p[1])] for p in self.pairs]}
