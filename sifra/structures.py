class XYPairs(object):
    def __init__(self, pairs):
        self.pairs = pairs

    def __iter__(self):
        return iter(self.pairs)

    def __jsonify__(self, *args, **kwargs):
        return {
            'class': [type(self).__module__, type(self).__name__],
            'pairs': [[float(p[0]), float(p[1])] for p in self.pairs]}

