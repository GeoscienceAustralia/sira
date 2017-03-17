def get_all_subclasses(cls):
    clss = cls.__subclasses__()
    if clss:
        return ['{}.{}'.format(c.__module__, c.__name__) for c in clss] + \
        reduce(lambda x, y: x + get_all_subclasses(y), clss, [])
    else:
        return []

