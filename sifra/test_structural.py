import unittest as ut

# these are required for defining the data model
from sifra.structural import (
    CouchSerialisationProvider,
    Element,
    ValidationError,
    MultipleBasesOfTypeBaseError,
    AlreadySavedException,
    jsonify,
    generate_element_base)

from sifra.elements import (
    Model,
    Component,
    ResponseModel)


class Unreachable_from_elements(object):
    """
    Since :py:class:`ResponseModel` (which is the base class of
    :py:class:`StepFunc`) is defined in a different module using a
    different :py:class:`sifra.structural._Base`, we want to make sure that
    we can still instantiate classes that are not visible within that module (
    like this one) outside of that module.
    """

    def __jsonify__(self, flatten):
        return {'class': [type(self).__module__, type(self).__name__]}



class StepFunc(ResponseModel):
    xs = Element('list', 'X values for steps', Element.NO_DEFAULT,
        [lambda x: [float(val) for val in x]])
    ys = Element('list', 'Y values for steps', Element.NO_DEFAULT)
    dummy = Element(
        'Unreachable_from_elements',
        'Uncreachable from elements',
        Unreachable_from_elements)

    def __validate__(self):
        if len(self.xs) != len(self.ys):
            raise ValidationError('length of xs and ys must be equal')

    def __call__(self, value):
        for x, y in zip(self.xs, self.ys):
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




COUCH_URL = 'http://couch:5984'
DB_NAME = 'models'
provider = CouchSerialisationProvider(COUCH_URL, DB_NAME)
Base = generate_element_base(provider)



class Test1(ut.TestCase):
    def setUp(self):
        self.model = Model()
        frag_curve = StepFunc(xs=[1,2,3], ys=[0.,.5,1.])
        boiler = Component(frag_func=frag_curve)
        turbine = Component(frag_func = LogNormalCDF(median=0.1, beta=0.5))
        self.model.add_component('boiler', boiler)
        self.model.add_component('turbine', turbine)

    def tearDown(self):
        provider.delete_db()

    def test_can_call(self):
        """
        Test that a fragility function can be called after a model has been
        serialised and deserialised.
        """

        abscissa = 1.0
        object_id = 'my-instance'

        def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
            return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

        res_1 = self.model.components['turbine'].frag_func(abscissa)
        self.model.save(False, object_id)

        model_copy = Model.load(object_id)
        res_2 = model_copy.components['turbine'].frag_func(abscissa)

        self.assertTrue(isclose(res_1, res_2, abs_tol=1e-09))

    def test_to_from_json_like(self):
        """
        Test that a model can be created from one converted 'to JSON'.
        """

        model2 = Base.to_python(jsonify(self.model, False))

    def test_modifiability(self):
        """
        Test that:

            - a previously saved model cannot be modified, and
            - a freshly cloned model can be modified.
        """

        # first use the db provider directly. note that when we used the db
        # directly, it would be possible to save a previously saved db again
        # as there is no check for this (which is done in _Base.save())
        _id, _rev = provider.get_db().save(jsonify(self.model, False))
        model2 = Base.to_python(provider.get_db().get(_id))

        # check that we cannot modify the copy pulled from the db.
        with self.assertRaises(TypeError):
            model2.name = 'new name'

        # now use a model which has had save called on it
        model3 = model2.clone()
        # check that we can modify it at first
        model3.name = 'new name'
        # check that once it has been saved, it can no longer be modified
        model3.save(False)
        with self.assertRaises(TypeError):
            model3.name = 'new new name'

    def test_cannot_resave(self):
        """
        Check that a model which has been saved cannot be saved again.
        """

        nextVersionOfModel = self.model.save(False)
        with self.assertRaises(AlreadySavedException):
            self.model.save(False)

    def test_correct_hasattr(self):
        """
        Check that the method for checking existence of an attribute on an
        an instance excluding is predecessor is working.
        """

        self.model.thingy = 'hi'
        new_model = self.model.clone()
        self.assertFalse(new_model._hasattr('thingy'))

    def test_non_modified_serialisation(self):
        """
        When a model is saved and the resulting clone is saved without
        modification, the id of its predecessor should be the same as what it
        was cloned from and the result of jsonify should be length 1 (that is,
        the result should only contain the class string).
        """

        m1 = self.model.save(False)
        mj = jsonify(m1, False)
        self.assertEqual(len(mj), 1, "Result of jsonify should have had length 1")

        m2 = m1.save(False)
        self.assertEqual(m1.predecessor._id, m2.predecessor._id,
            "Models should have had same predecessor ids")



class Test2(ut.TestCase):
    def test_cannot_have_fields(self):
        """
        Check that we cannot create a model containing elements with
        dissallowed names.
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

