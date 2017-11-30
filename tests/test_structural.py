import unittest as ut

# these are required for defining the data model
from sifra.modelling.structural import (
    Base,
    Element,
    jsonify,
    MultipleBasesOfTypeBaseError,
    pythonify)

from modelling.responsemodels import ResponseModel
from sifra.modelling.component import Component
from sifra.modelling.infrastructure_system import IFSystem

from sifra.modelling.structures import (
    XYPairs)

from sifra.settings import SQLITE_TEST_DB_FILE

# TODO setup a test database
DB_CONNECTION_STRING = 'sqlite:///{}'.format(SQLITE_TEST_DB_FILE)


class Unreachable_from_elements(object):
    """
    Since :py:class:`ResponseModel` (which is the base class of
    :py:class:`StepFunc`) is defined in a different module using a
    different :py:class:`sifra.structural._Base`, we want to make sure that
    we can still instantiate classes that are not visible within that module (
    like this one) outside of that module.
    """

    def __jsonify__(self):
        return {'class': [type(self).__module__, type(self).__name__]}


class StepFunc(ResponseModel):
    xys = Element('XYPairs', 'A list of X, Y pairs.', list,
        [lambda xy: [(float(x), float(y)) for x, y in xy]])

    dummy = Element(
        'Unreachable_from_elements',
        'Unreachable from elements',
        Unreachable_from_elements)

    def __call__(self, value):
        """
        Note that intervals are closed on the right.
        """
        for x, y in self.xys:
            if value < x:
                return y

        raise ValueError('value is greater than all xs!')


class LogNormalCDF(ResponseModel):
    median = Element('float', 'Median of the log normal CDF.',
            Element.NO_DEFAULT, [lambda x: float(x) > 0.])
    beta = Element('float', 'Log standard deviation of the log normal CDF',
            Element.NO_DEFAULT, [lambda x: float(x) > 0.])

    def __call__(self, value):
        import scipy.stats as stats
        return stats.lognorm.cdf(value, self.beta, scale=self.median)


class Test1(ut.TestCase):
    def setUp(self):
        self.model = IFSystem(name="model_test")
        frag_curve = StepFunc(xys=XYPairs([[1., 0.], [2., .5], [3., 1.]]))

        boiler_dict = {'frag_func': frag_curve,
                       'operating_capacity': 1.0,
                       'component_type': 'Boiler',
                       'component_class': 'Boiler System',
                       'destination_components': ['precip_1a', 'precip_1b'],
                       'node_cluster': 'Boiler System',
                       'cost_fraction': 0.1443,
                       'node_type': 'transshipment',
                       'recovery_func': [3.0, 0.61, 1.0]}

        boiler = Component(**boiler_dict)

        frag_func = LogNormalCDF(median=0.1, beta=0.5)
        turbine_dict = {'frag_func': frag_func,
                        'operating_capacity': 1.0,
                        'component_type': 'Turbine and Condenser',
                        'component_class': 'Condenser',
                        'destination_components': ['gen_1', 'gen_2'],
                        'node_cluster': 'Condenser',
                        'cost_fraction': 0.1443,
                        'node_type': 'transshipment',
                        'recovery_func': [5.0, 0.61, 1.0]}

        turbine = Component(**turbine_dict)
        self.model.add_component('boiler', boiler)
        self.model.add_component('turbine', turbine)

    def tearDown(self):
        Base._provider.delete_db()

    def test_can_call(self):
        """
        Test that a fragility function can be called after a model has been
        serialised and deserialised.
        """

        abscissa = 1.0
        object_name = 'my-instance'

        def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
            return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

        res_1 = self.model.components['turbine'].frag_func(abscissa)
        oid = self.model.save()

        model_copy = IFSystem.load(oid)
        name = value = None
        for name, value in model_copy._attributes.iteritems():
            if name == 'name':
                break

        self.assertIsNotNone(name, 'attribute "name" should not be None.')
        self.assertEqual(value, object_name)

        res_2 = model_copy.components['turbine'].frag_func(abscissa)

        self.assertTrue(isclose(res_1, res_2, abs_tol=1e-09))

    def test_to_from_json_like(self):
        """
        Test that a model can be created from one converted 'to JSON'.
        """

        model2 = pythonify(jsonify(self.model))

    def test_jsonify_with_metadata(self):
        """
        Test that :py:meth:`sifra.structural.Base.jsnoify_with_metadata` does
        not raise an exception. This test needs to do more.
        """

        data = self.model.jsonify_with_metadata()


class Test2(ut.TestCase):
    def test_cannot_have_fields(self):
        """
        Check that we cannot create a model containing elements with
        dissallowed names, such as "predecessor".
        """

        with self.assertRaises(ValueError):
            cls = type(
                'Tst',
                (Base,),
                {'predecessor': Element('object', 'dissallowed name', object)})

    def test_single_base_of_type_base(self):
        """
        Check that a model cannot inherit from Base more than once.
        """

        c1 = type('C1', (Base,), {})
        c2 = type('C2', (Base,), {})
        with self.assertRaises(MultipleBasesOfTypeBaseError):
            c3 = type('C3', (c1, c2), {})


class Test3(ut.TestCase):
    def test_gets_all_subclasses(self):
        sc_names = [cls.__name__ for cls in ResponseModel.__subclasses__()]

        for nm in ('StepFunc', 'LogNormalCDF'):
            self.assertIn(nm, sc_names)


class Test4(ut.TestCase):
    def test_has_json_desc(self):
        desc = IFSystem.__json_desc__
        self.assertIn('description', desc, 'Model should contain "description"')
        self.assertIn('components', desc, 'Model should cotain "components"')


if __name__ == '__main__':
    ut.main()
