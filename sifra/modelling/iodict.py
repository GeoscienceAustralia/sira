from collections import OrderedDict

from sifra.modelling.utils import jsonify

from sifra.modelling.structural import Base


class IODict(OrderedDict, Base):
    def __init__(self, *args, **kwargs):
        super(IODict, self).__init__(*args, **kwargs)
        if 'key_index' in kwargs:
            key_index = kwargs.pop('key_index')
            self.key_index = {i: k for i, k in enumerate(key_index.iterkeys())}
        else:
            self.key_index = {i: k for i, k in enumerate(self.iterkeys())}

    def __setitem__(self, key, value):
        super(IODict, self).__setitem__(key, value)
        self.key_index = {i: k for i, k in enumerate(self.iterkeys())}

    def index(self, index):
        return super(IODict, self).__getitem__(self.key_index[index])

    def __jsonify__(self):
        """
        Validate this instance and transform it into an object suitable for
        JSON serialisation.
        """
        res = {
            jsonify(k): jsonify(v)
            for k, v in self.iteritems()}
        res['class'] = [type(self).__module__, type(self).__name__]
        res['key_index'] = self.key_index
        return res

    @classmethod
    def __pythonify__(cls, val):
        return IODict(val)


