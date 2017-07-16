from copy import deepcopy
from collections import namedtuple
from itertools import izip, imap

def get_all_subclasses(cls):
    clss = cls.__subclasses__()
    if clss:
        return ['{}.{}'.format(c.__module__, c.__name__) for c in clss] + \
        reduce(lambda x, y: x + get_all_subclasses(y), clss, [])
    else:
        return []



def _make_diff(name, elements):
    def __new__(cls, *args):
        if all(imap(lambda x: x is None, args)):
            return None
        return super(cls, cls).__new__(cls, *args)
    return type(name, (namedtuple(name, elements),), {'__new__': __new__})

Diff = _make_diff('Diff', ['changed'])
DictDiff = _make_diff('DictDiff', ['changed', 'dropped', 'added'])
ListDiff = _make_diff('ListDiff', ['changed', 'dropped', 'added'])



def find_changes(old, new):
    if old == new:
        return None

    if type(new) is tuple: # deliberately not isinstance... who knows what we could throw away
        new = list(new)

    if type(old) is tuple: # deliberately not isinstance... who knows what we could throw away
        old = list(old)

    if old == new:
        return None

    if type(old) != type(new):
        return Diff(new or None)

    if type(new) == list:
        all_changes = {}
        for a, b, index in izip(old, new, range(len(new))):
            changes = find_changes(a, b)
            if changes:
                all_changes[index] = changes

        if len(new) < len(old):
            return ListDiff(all_changes or None, len(old)-len(new), None)
        if len(new) > len(old):
            return ListDiff(all_changes or None, None, new[len(old):])
        else:
            assert all_changes, 'should not have got here'
        return ListDiff(all_changes or None, None, None)

    if type(new) != dict:
        return Diff(new or None)

    new_keys = set(new.iterkeys())
    old_keys = set(old.iterkeys())
    dropped_keys = old_keys - new_keys

    added = {k: new[k] for k in new_keys - old_keys}

    changed = {}
    for k in new_keys & old_keys:
        changes = find_changes(old[k], new[k])
        if changes:
            changed[k] = changes

    return DictDiff(changed or None, dropped_keys or None, added or None)



def reconstitute(old, changes):
    result = list(old) if type(old) == tuple else deepcopy(old)

    if isinstance(changes, Diff):
        result = changes.changed

    elif isinstance(changes, DictDiff):
        assert isinstance(old, dict)

        if changes.dropped is not None:
            for k in changes.dropped:
                result.pop(k)

        if changes.added is not None:
            result.update(changes.added)

        if changes.changed is not None:
            for k, v in changes.changed.iteritems():
                result[k] = reconstitute(old[k], v)

    else:
        assert isinstance(changes, ListDiff)
        assert type(old) == list or type(old) == tuple # using type (not is instance) deliberately

        if changes.dropped is not None:
            for k in changes.dropped:
                result = result[:-changes.dropped]

        if changes.added is not None:
            result += changes.added

        if changes.changed is not None:
            for k, v in changes.changed.iteritems():
                result[k] = reconstitute(old[k], v)

    return result
