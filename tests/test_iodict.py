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

    def test_list_init(self):
        # create a string list of values
        dser_vals = [("{:02d}".format(v), chr(80-v)) for v in range(33, 0, -1)]
        # create the io dict from the string
        io_deser = IODict.__pythonify__(dser_vals)

        # check length
        self.assertTrue(len(io_deser) == 33)

        # check that the iteration order is the order of insertion
        for listy, ioey in zip(dser_vals, io_deser):
            self.assertEquals(listy[0], ioey)

        # check that the index references the correct value in list
        for index, val in enumerate(dser_vals):
            self.assertEquals(io_deser.index(index), val[1])

    def test_dict_init(self):
        # create a string list of values
        dser_vals = {"{:02d}".format(v): chr(80-v) for v in range(33, 0, -1)}
        key_index = {x[0]: x[1] for x in enumerate(reversed(sorted(dser_vals.keys())))}

        # create the io dict from the string
        dser_vals['key_index'] = key_index
        io_deser = IODict.__pythonify__(dser_vals)

        # check length
        self.assertTrue(len(io_deser) == 33)

        # check that the key_index order is correct
        for index_key, io_key in zip(key_index.keys(), io_deser.keys()):
            self.assertEquals(key_index[index_key], io_key)
            self.assertEquals(io_deser.index(index_key), io_deser[io_key])

        # check that the index references the correct value in list
        for index, val in enumerate(dser_vals):
            self.assertEquals(io_deser.index(index), val[1])

