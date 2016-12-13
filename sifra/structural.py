import abc
import collections



DB_NAME = 'models'



class NoDefaultException(Exception):
    """
    Thrown when a :py:class:`_Base` is created without providing values for one
    or more :py:class:`Element`s which do not have default values.

    Note that users should never be instantiating or subclassing
    :py:class:`_Base` directly. One should extend a class returned by
    :py:func:`generate_element_base`, which returns a class which extends
    :py:class:`_Base`.
    """
    pass

class ValidationError(Exception):
    """
    Thrown when validation of some item fails. Some examples of when this may
    occur are:

        - A value for an :py:class:`Element` is provided which is not an
          an instance of the type specified for the Element (which is specified
          via argument *cls* to :py:meth:`Element.__init__`).

        - One of the validators provided for an element (see the agument
          *validators* to :py:class:`Element.__init__`) fails or raises an
          exception of this type.
    """
    pass

class AlreadySavedException(Exception):
    """
    Raised if an attempt is made to save a 'Document' which has previously been
    saved.
    """
    pass

class DisallowedElementException(ValueError):
    """
    Raised if an an attempt is made to define an element with a dissallowed
    name. Dissallowed names are specified by
    :py:attr:StructuralMeta.DISALLOWED_FIELDS.
    """
    pass

class MultipleBasesOfTypeBaseError(ValueError):
    """
    Raised if an attempt is made to define a class which inherits from
    multiple classes (``c``) for which ``issubclass(type(c), StructuralMeta)``
    is *True*.

    The reason to dissalow multiple inheritance of said classes is to conform
    to the structure of XML, where an element can only have one parent. This may
    not turn out to be an issue as other interpretations of class hierachies in
    the context of XML may be sensible/feasible... but the semantics and
    practicalities would need to be considered so stop this for now and see how
    we go.
    """
    pass



def jsonify(obj, flatten):
    """
    Convert an object to a representation that can be converted to a JSON document.

    The algorithm is:

        - if the object has a method ``__jsonify__``, return the result of calling it, otherwise
        - if ``isinstance(obj, dict)``, transform the key/value (``k:v``) pairs with
            ``jsonify(k): jsonify(v)``, otherwise
        - if the object is iterable but not a string, transform the elements (``v``)
            with ``jsonify(v)``.
    """

    if hasattr(obj, '__jsonify__'):
        # should probably check the number of args
        return obj.__jsonify__(flatten)
    if isinstance(obj, dict):
        return {jsonify(k, flatten) : jsonify(v, flatten) for k, v in obj.iteritems()}
    if isinstance(obj, collections.Iterable) and not isinstance(obj, basestring):
        return [jsonify(v, flatten) for v in obj]
    return obj



def to_python(obj):
    """
    Convert a 'jsonified' object to a Python object. This is the inverse of
    :py:func:`jsonify`.

    A python instance ``c`` should be eqivalent to ``to_python(jsonify(c))``
    with respect to the data returned by ``__jsonify__`` if the object has that
    method or the object as a whole otherwise.
    """

    if isinstance(obj, dict):
        if 'class' in obj:
            cls = obj.pop('class')
            return eval(cls)(**to_python(obj))
        return {str(k): to_python(v) for k, v in obj.iteritems()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj



class Element(object):
    """
    Represents an element of a model. If a model were represented in a relational
    database, this is analgous to a field in a table.
    """

    @staticmethod
    def NO_DEFAULT():
        """
        A callable that can be used to signal that an Element has no default
        value. Simply raises a :py:exception:`NoDefaultException`.
        """

        raise NoDefaultException()

    def __init__(self, cls, description, default=None, validators=None):
        self.cls = cls
        self.description = description
        self._default = default
        self.validators = validators

    @property
    def default(self):
        if self._default is False:
            raise NoDefaultException()
        return self._default() if callable(self._default) else self._default

    def __jsonify__(self, val, flatten):
        """
        Convert *val* to a form that can be json sersialised.
        """

        self.__validate__(val)
        return jsonify(val, flatten)

    def __validate__(self, val):
        """
        Validate *val*. This checks that *val* is of subclass ``eval(self.cls)``
        and that no :py:attr:`validators` either return *False* or raise
        exceptions.

        :py:raises:`ValidationError`.
        """

        if not isinstance(val, eval(self.cls)):
            raise ValidationError('value is not instance of {}'.format(self.cls))

        if self.validators is not None:
            for v in self.validators:
                try:
                    if v(val) is False:
                        raise ValidationError('validator {} returned False'.format(str(v)))
                except ValidationError as e:
                    raise e
                except Exception as e:
                    raise ValidationError(str(e))



class StructuralMeta(type):
    #: Names of :py:class:`Element`s that cannot defined on any class ``c`` for
    #: which ``issubclass(type(c), StructuralMeta)`` is *True*. These are names
    #: of elements which are used internally and for the sake of the performance
    #: of attribute lookup, are banned for other use.
    DISALLOWED_FIELDS = ['class', 'predecessor', '_id', '_rev', '_provider']

    def __new__(cls, name, bases, dct):
        # check that only one base is instance of _Base
        if len([base for base in bases if issubclass(type(base), StructuralMeta)]) > 1:
            raise MultipleBasesOfTypeBaseError('Invalid bases in class {}'.format(name))

        # extract the parameters
        params = {}
        for k in dct.keys():
            if isinstance(dct[k], Element):
                params[k] = dct.pop(k)

        # cannot have a parameter with name class, as this messes with
        # serialisation
        for field in StructuralMeta.DISALLOWED_FIELDS:
            if field in params:
                raise DisallowedElementException(
                    'class {} cannot have Element with name "{}"'.format(name, field))

        dct['__params__'] = params

        return super(StructuralMeta, cls).__new__(cls, name, bases, dct)



class _Base(object):
    """
    Base class for all 'model' classes. **This should never be used by clients**
    and serves as a base class for dynamically generated classes returned by
    :py:func:`generate_element_base`, which are designed for use by clients.
    """

    __metaclass__ = StructuralMeta

    def __init__(self, **kwargs):
        # can't do the following with self._id, as this causes problems with
        # __setattr__ and, in particular, __getattr__.
        object.__setattr__(self, '_id', None)
        _id = kwargs.pop('_id', None)
        self._rev = kwargs.pop('_rev', None)
        self._predecessor = kwargs.pop('predecessor', None)

        if self._predecessor is None:
            # then we provide default values for each element
            for k, v in self.__params__.iteritems():
                if k not in kwargs:
                    try:
                        kwargs[k] = v.default
                    except NoDefaultException:
                        raise ValueError('Must provide value for {}'.format(k))
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

        # must be done last to avoid __setattr__ getting upset.
        if _id is not None:
            self._id = _id

    def __setattr__(self, name, value):
        """
        Override of :py:meth:`object.__setattr__` which raises
        :py:exception:`TypeError` if an attempt is made to set an attribute on
        an instance for which has already been saved.

        .. overrides::`object.__setattr__`

        """

        if self._id is not None:
            raise TypeError('Cannot modify saved item. Please clone first.')
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        """
        Get an attribute from the objects predecessor.
        """

        try:
            return getattr(self.predecessor, name)
        except AttributeError:
            raise AttributeError("'{}' has no attribute {}".format(self.__class__.__name__, name))

    @property
    def predecessor(self):
        """
        The objects predecessor. This goes back to the database if required.
        """

        if isinstance(self._predecessor, basestring):
            self._predecessor = to_python(self.db.get(self._predecessor))
        return self._predecessor

    @property
    def db(self):
        """
        Get the db to be used by this instance.

        .. todo:: This is currently not thread safe... fix this.
        """

        return self._provider.get_db_for(DB_NAME)

    def __validate__(self):
        """
        Validate this instance.
        """

        pass

    def _hasattr(self, key):
        # like hasattr, but does not look into predecessor.
        try:
            object.__getattribute__(self, key)
        except AttributeError:
            return False
        return True

    def __jsonify__(self, flatten):
        """
        Validate this instance and transform it into an object suitable for
        JSON serialisation.
        """
        hasa = lambda k: hasattr(self, k) if flatten else self._hasattr

        self.__validate__()
        res = {'class': type(self).__name__}
        res.update({
            jsonify(k, flatten): v.__jsonify__(getattr(self, k), flatten)
            for k, v in self.__params__.iteritems()
            if hasa(k)})
        return res

    def clone(self):
        """
        Clone this instance. This creates and returns a new instance with
        predecessor *self*.
        """

        return self.__class__(predecessor=self)

    def save(self, flatten):
        """
        Save this instance.
        """

        if self._id is not None:
            # then this has been saved before!
            raise AlreadySavedException('Document has already been saved.')

        res = jsonify(self, flatten)

        if self._predecessor is not None:
            if isinstance(self._predecessor, basestring):
                res['predecessor'] = self._predecessor
            elif hasattr(self._predecessor, '_id'):
                res['predecessor'] = self._predecessor._id

        # cannot do the following in one line as we need to set self._id last
        doc = self.db.save(res)
        self._rev = doc[1]
        self._id = doc[0]

        return self.clone()



def generate_element_base(provider):
    """
    Generate a base class for deriving 'model' classes from.

    :param provider: Serialisation provider to get 'database connections' from.
    :type provider: :py:class:`SerialisationProvider`
    """

    return type('ElementBase', (_Base,), {'_provider': provider})



class SerialisationProvider(object):
    """
    Provides access to an object that can be used to serialise models or other
    components.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_db_for(self, db_name):
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_db(self, b_name):
        raise NotImplementedError()



class CouchSerialisationProvider(SerialisationProvider):
    COUCH_URL = 'http://couch:5984'

    def __init__(self):
        import couchdb
        self._server = couchdb.Server(CouchSerialisationProvider.COUCH_URL)

    def get_db_for(self, db_name):
        import couchdb
        try:
            return self._server[db_name]
        except couchdb.http.ResourceNotFound:
            return self._server.create(db_name)

    def delete_db(self, db_name):
        import couchdb
        try:
            self._server.delete(db_name)
        except couchdb.ResourceNotFound:
            pass



if __name__ == "__main__":
    # classes used for testing
    provider = CouchSerialisationProvider()
    Base = generate_element_base(provider)

    class Model(Base):
        components = Element('dict', 'A component', dict,
            [lambda x: [isinstance(y, Component) for y in x.itervalues()]])

        name = Element('str', "The model's name", 'model')

        def add_component(self, name, component):
            self.components[name] = component



    class Component(Base):
        frag_func = Element('ResponseModel', 'A fragility function', Element.NO_DEFAULT)

        def expose_to(self, pga):
            return self.frag_func(pga)



    class ResponseModel(Base):
        def __call__(self, pga):
            raise NotImplementedError()



    class StepFunc(ResponseModel):
        xs = Element('list', 'X values for steps', Element.NO_DEFAULT,
            [lambda x: [float(val) for val in x]])
        ys = Element('list', 'Y values for steps', Element.NO_DEFAULT)

        def __validate__(self):
            if len(self.xs) != len(self.ys):
                raise ValidationError('length of xs and ys must be equal')

        def __call__(self, value):
            for x, y in zip(self.xs, self.ys):
                if value < x:
                    return y

            raise ValueError('value is greater than all xs!')



    from pprint import pprint
    import unittest as ut

    class Test1(ut.TestCase):
        def setUp(self):
            self.model = Model()
            frag_curve = StepFunc(xs=[1,2,3], ys=[0.,.5,1.])
            boiler = Component(frag_func=frag_curve)
            self.model.add_component('boiler', boiler)

        def tearDown(self):
            provider.delete_db(DB_NAME)

        def test_to_from_json_like(self):
            """
            Test that a model can be created from one converted 'to json'.
            """

            model2 = to_python(jsonify(self.model, False))

        def test_modifiability(self):
            """
            Test that a previously save model cannot be modified.
            """

            # first use the db provider directly
            _id, _rev = provider.get_db_for(DB_NAME).save(jsonify(self.model, False))
            model2 = to_python(provider.get_db_for(DB_NAME).get(_id))
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


    ut.main()

