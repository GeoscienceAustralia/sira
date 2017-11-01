from collections import OrderedDict

from sifra.modelling.utils import jsonify, pythonify

from sifra.modelling.structural import Base


class IODict(OrderedDict, Base):
    """
    An indexable, ordered dictionary.
    The infrastructure components required a data structure that would maintain its order,
    and allowed key value as well as index value access.
    It also has to support _jsonify__ and __pythonify__ methods.
    """

    def __init__(self, *args, **kwargs):
        super(IODict, self).__init__(*args, **kwargs)
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
        new_io = IODict()

        if 'key_index' in val:
            key_index = val.pop('key_index')
            for comp_index in sorted(key_index.keys()):
                dict_name = key_index[comp_index]
                value = val[dict_name]
                new_io[dict_name] = pythonify(value)

        return new_io



