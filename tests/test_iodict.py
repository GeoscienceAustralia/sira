import unittest

from sifra.modelling.iodict import IODict


class TestIODict(unittest.TestCase):
    def test_init(self):
        first = IODict()
        self.assertTrue(len(first) == 0)

    def test_add_iter(self):
        #
        testy = IODict()
        testy[3] = 'z_three'
        testy[2] = 'y_two'
        testy[1] = 'x_one'

        for num, (key, value) in enumerate(testy.items()):
            self.assertTrue(num+key == 3)
            self.assertTrue(value == testy[key])
            self.assertTrue(value == testy.index(num))

    def test_deserialise(self):
        # TODO better test for deserialise
        dser_vals = [("{:02d}".format(v), chr(80-v)) for v in range(33, 0, -1)]

        deser = IODict.__pythonify__(dser_vals)
        # check length
        self.assertTrue(len(deser) == 33)

        for listy, ioey in zip(dser_vals, deser):
            self.assertEquals(listy, ioey)


