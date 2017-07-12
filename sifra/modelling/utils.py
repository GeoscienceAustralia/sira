from collections import OrderedDict

def get_all_subclasses(cls):
    clss = cls.__subclasses__()
    if clss:
        return ['{}.{}'.format(c.__module__, c.__name__) for c in clss] + \
        reduce(lambda x, y: x + get_all_subclasses(y), clss, [])
    else:
        return []


class IODict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(IODict, self).__init__(*args, **kwargs)
        self.key_index = {i: k for i, k in enumerate(self.iterkeys())}

    def __setitem__(self, key, value):
        super(IODict, self).__setitem__(key, value)
        self.key_index = {i: k for i, k in enumerate(self.iterkeys())}

    def index(self, index):
        return super(IODict, self).__getitem__(self.key_index[index])