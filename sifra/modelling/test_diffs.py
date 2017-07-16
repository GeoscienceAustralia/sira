import unittest
from utils import Diff, DictDiff, ListDiff, find_changes, reconstitute

"""
Tests of code for finding differences between two jsonifiable dictionaries.
"""

def tups2lists(obj):
    """
    Recursively convert any tuples found in *obj* to lists.
    """

    if isinstance(obj, dict):
        return {tups2lists(k): tups2lists(v) for k, v in obj.iteritems()}
    if isinstance(obj, tuple):
        return [tups2lists(v) for v in obj]
    return obj



class Tests(unittest.TestCase):
    def test1(self):
        a = {
            'a': {'a': 'a'},
            'b': 1,
            'c': ['a', 'a'],
            'd': ['a', 'a', 'a'],
            'e': ('a', 'a', 'a'),
            'f': ('a', 'a', 'a'),
            'g': ('a', 'a', 'a')}

        b = {
            'a': {'a': 'a'},
            'b': 2,
            'c': ['a', {'a': 'a'}],
            'd': ['a', 'a', 'a', 'a'],
            'e': ['a', 'a', 'a'],
            'f': ('a', 'a', 'a', 'a'),

            'h': ('a', 'a', 'a')}

        self.assertEqual(
            tups2lists(reconstitute(a, find_changes(a, b))),
            tups2lists(b))

    def test2(self):
        a = b = 42
        self.assertIs(find_changes(a, b), None)

    def test3(self):
        a = 1
        b = 2
        self.assertIsInstance(find_changes(a, b), Diff)

    def test4(self):
        a = {'a': 'a'}
        b = {'a': 'b'}
        self.assertIsInstance(find_changes(a, b), DictDiff)

    def test5(self):
        a = [1]
        b = [2]
        self.assertIsInstance(find_changes(a, b), ListDiff)

    def test6(self):
        class A(object):
            def __init__(self, val):
                self.val = val

            def __eq__(self, other):
                return self.val == other.val

        a = A(1)
        b = A(2)
        self.assertIsInstance(find_changes(a, b), Diff)

    def test7(self):
        class A(object): pass
        class B(object): pass

        a = A()
        b = B()
        diff = find_changes(a, b)
        self.assertIsInstance(diff, Diff)
        self.assertIsInstance(diff.changed, B)
