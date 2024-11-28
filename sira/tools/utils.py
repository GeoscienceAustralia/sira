"""
sira/tools/utils.py
This module provides a collection of helper functions
"""
from pathlib import Path
from functools import reduce
import importlib
from copy import deepcopy
from collections import namedtuple
from collections.abc import Iterable
import os


def relpath(self, start=None):
    return Path(os.path.relpath(self, start))


def get_all_subclasses(cls):
    clss = cls.__subclasses__()
    if clss:
        return ['{}.{}'.format(c.__module__, c.__name__) for c in clss] + \
            reduce(lambda x, y: x + get_all_subclasses(y), clss, [])
    else:
        return []


def jsonify(obj):
    """
    Convert an object to a representation that can be converted to a JSON
    document.

    The algorithm is:

    - if the object has a method ``__jsonify__``, return the result of
        calling it, otherwise
    - if ``isinstance(obj, dict)``, transform the key/value (``k:v``) pairs
        with ``jsonify(k): jsonify(v)``, otherwise
    - if the object is iterable but not a string, transform the elements (``v``)
        with ``jsonify(v)``.
    """

    if hasattr(obj, '__jsonify__'):
        # should probably check the number of args, or change the name of the
        # two arg version
        return obj.__jsonify__()
    if isinstance(obj, dict):
        return {jsonify(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, Iterable) and not isinstance(obj, str):
        return [jsonify(v) for v in obj]

    return obj


def pythonify(obj):
    """
    Convert a 'jsonified' object to a Python object. This is the inverse of
    :py:func:`jsonify`.

    A python instance ``c`` should be eqivalent to ``__call__(jsonify(c))``
    with respect to the data returned by ``__jsonify__`` if the object has that
    method or the object as a whole otherwise.
    """

    if isinstance(obj, dict):
        attrs = obj.pop('_attributes', None)
        if 'class' in obj:
            cls = obj.pop('class')
            clazz = class_getter(cls)
            if hasattr(clazz, '__pythonify__'):
                res = clazz.__pythonify__(obj)
            else:
                res = class_getter(cls)(**pythonify(obj))
        else:
            res = {str(k): pythonify(v) for k, v in obj.items()}
        # pylint: disable=protected-access
        if attrs is not None:
            res._attributes = attrs
        return res
    if isinstance(obj, list):
        return [pythonify(v) for v in obj]
    return obj


def class_getter(mod_class):
    return getattr(importlib.import_module(mod_class[0]), mod_class[1])


def _make_diff(name, elements):
    def __new__(cls, *args, **kwargs):
        if len(kwargs):
            args += tuple((kwargs.get(e, None) for e in elements[len(args):]))
        if all(map(lambda x: x is None, args)):
            return None
        return super(cls, cls).__new__(cls, *args)

    def __jsonify__(self):
        res = {
            e: jsonify(getattr(self, e))
            for e in elements if getattr(self, e) is not None
        }
        res['class'] = [type(self).__module__, type(self).__name__]
        return res

    return type(
        name,
        (namedtuple(name, elements),),
        {'__new__': __new__, '__jsonify__': __jsonify__})


Diff = _make_diff('Diff', ['changed'])
DictDiff = _make_diff('DictDiff', ['changed', 'dropped', 'added'])
ListDiff = _make_diff('ListDiff', ['changed', 'dropped', 'added'])


def find_changes(old, new):
    if isinstance(new, tuple):
        # deliberately not isinstance... who knows what we could throw away
        new = list(new)

    if isinstance(old, tuple):
        # deliberately not isinstance... who knows what we could throw away
        old = list(old)

    if old == new:
        return None

    if type(old) is not type(new):
        return Diff(new or None)

    if isinstance(new, list):
        all_changes = {}
        for a, b, index in zip(old, new, range(len(new))):
            changes = find_changes(a, b)
            if changes:
                all_changes[index] = changes

        if len(new) < len(old):
            return ListDiff(all_changes or None, len(old) - len(new), None)
        if len(new) > len(old):
            return ListDiff(all_changes or None, None, new[len(old):])
        else:
            assert all_changes, 'should not have got here'

        return ListDiff(all_changes or None, None, None)

    if not isinstance(new, dict):
        return Diff(new or None)

    new_keys = set(new.keys())
    old_keys = set(old.keys())
    dropped_keys = old_keys - new_keys

    added = {k: new[k] for k in new_keys - old_keys}

    changed = {}
    for k in new_keys & old_keys:
        changes = find_changes(old[k], new[k])
        if changes:
            changed[k] = changes

    return DictDiff(changed or None, dropped_keys or None, added or None)


def reconstitute(old, changes):
    result = list(old) if isinstance(old, tuple) else deepcopy(old)

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
            for k, v in changes.changed.items():
                result[k] = reconstitute(old[k], v)

    else:
        assert isinstance(changes, ListDiff)
        assert isinstance(old, list) or isinstance(old, tuple)
        # using type (not is instance) deliberately

        if changes.dropped is not None:
            for k in changes.dropped:
                result = result[:-changes.dropped]

        if changes.added is not None:
            result += changes.added

        if changes.changed is not None:
            for k, v in changes.changed.items():
                result[int(k)] = reconstitute(old[int(k)], v)

    return result


def wrap_file_path(
        file_path,
        max_width=85,
        first_line_indent=" " * 9,
        subsequent_indent=" " * 9):
    # Replace backslashes with forward slashes for consistent splitting
    file_path = file_path.replace('\\', '/')
    path_components = file_path.split('/')

    lines = []
    current_line = first_line_indent
    for i, component in enumerate(path_components):
        # If this isn't the first component, then add a slash before it
        if i > 0:
            # Check if adding slash + component would exceed max_width
            if len(current_line) + len(component) + 1 > max_width:
                # If so, start a new line
                lines.append(current_line)
                current_line = subsequent_indent + '/' + component
            else:
                # Otherwise, add slash and component to current line
                current_line += '/' + component
        else:
            # For the first component, just add it
            # (it might be an empty string if path starts with '/')
            current_line += component

    # Add the last line
    if current_line:
        lines.append(current_line)

    txtblock = '\n'.join(lines)
    if '\\' in file_path:
        txtblock = txtblock.replace('/', '\\')

    return txtblock
