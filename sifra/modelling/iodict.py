from collections import OrderedDict

from sifra.modelling.utils import jsonify, pythonify

from sifra.modelling.structural import Base


class IODict(OrderedDict, Base):
    """
    An indexable, ordered dictionary.
    The infrastructure components required a data structure that maintains its order,
    and also provides key and index value access.
    It must also support the _jsonify__ and __pythonify__ methods.
    """

    def __init__(self, *args, **kwargs):
        """Build the key index after initialisation"""
        super(IODict, self).__init__(*args, **kwargs)
        self.key_index = {i: k for i, k in enumerate(self.iterkeys())}

    def __setitem__(self, key, value):
        """Build the key index after adding a new item."""
        super(IODict, self).__setitem__(key, value)
        self.key_index = {i: k for i, k in enumerate(self.iterkeys())}

    def index(self, index):
        """
        Return an item using the key_index values
        :param index: The offset of the established order.
        :return: Item at the parameter offset.
        """
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
        """
        Convert the parameter into an IODict. Using
        the key_index value, if present, to preserve
        the desired order.

        :param val: A map or iterable.
        :return: IODict
        """
        if isinstance(val, dict):
            if 'key_index' in val:
                # Create an IODict with the supplied order
                new_io = IODict()
                key_index = val.pop('key_index')
                for comp_index in sorted(key_index.keys()):
                    dict_name = key_index[comp_index]
                    value = val[dict_name]
                    new_io[dict_name] = pythonify(value)

                return new_io
            else:
                # Construct the dictionary and create a new order
                return IODict(**val)
        else:
            # Construct the dict with the order supplied by the iterable
            return IODict(val)





