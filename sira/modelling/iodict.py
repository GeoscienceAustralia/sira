from collections import OrderedDict
from sira.modelling.structural import Base


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
        self.key_index = {i: k for i, k in enumerate(self.keys())}

    def __setitem__(self, key, value):
        """Build the key index after adding a new item."""
        super(IODict, self).__setitem__(key, value)
        self.key_index = {i: k for i, k in enumerate(self.keys())}

    def index(self, index):
        """
        Return an item using the key_index values
        :param index: The offset of the established order.
        :return: Item at the parameter offset.
        """
        return super(IODict, self).__getitem__(self.key_index[index])





