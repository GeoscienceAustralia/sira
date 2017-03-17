import abc
import importlib
import collections
import inspect
from copy import deepcopy
from sifra.settings import USE_COUCH_DB, COUCH_DB_URL, COUCH_DB_NAME
from sifra.modelling.serialisation import SerialisationProvider



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
    Raised if an an attempt is made to define an element with a disallowed
    name. Disallowed names are specified by
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



def class_getter(mod_class):
    return getattr(importlib.import_module(mod_class[0]), mod_class[1])



def jsonify(obj):
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
        return obj.__jsonify__()
    if isinstance(obj, dict):
        return {jsonify(k) : jsonify(v) for k, v in obj.iteritems()}
    if isinstance(obj, collections.Iterable) and not isinstance(obj, basestring):
        return [jsonify(v) for v in obj]
    return obj



def _merge_data_and_metadata(meta, data):
    """
    .. note:: It is pretty inefficient duplicating the data as much as we do in
        this method. It would be possible to do this based on the value alone
        (getting the json_desc as required) in the browser. The only trick is,
        being able to differentiate between a list and a dict in javascript.
    """

    if not isinstance(data, dict):
        meta['_value'] = data
        return

    if 'class' in data and meta['class'] != '.'.join(data['class']):
        if not issubclass(
            class_getter(data['class']),
            class_getter(meta['class'].rsplit('.', 1))):
            raise Exception('inconsisent class structure')
        meta.update(deepcopy(class_getter(data['class']).__json_desc__))

    meta['_value'] = data
    for k, v in data.iteritems():
        if k == 'class' or k not in meta:
            continue
        if isinstance(v, dict):
            _merge_data_and_metadata(meta[k], v)
            if meta[k]['class'] == '__builtin__.dict':
                meta[k]['_items'] = {}
                for k1, v1 in v.iteritems():
                    # note that the following line implys that dicts can
                    # only contain classes that extend sifra.structural.Base, or
                    # at least have __json_desc__ defined
                    nextMeta = deepcopy(class_getter(v1['class']).__json_desc__)
                    meta[k]['_items'][k1] = nextMeta
                    _merge_data_and_metadata(nextMeta, v1)
        elif isinstance(v, list):
            # check that meta thinks we are dealing with a dict... should
            # be debug time assert when happy with this.
            if meta[k]['class'] != '__builtin__.list':
                raise Exception('inconsistent value and description')
            meta[k]['_value'] = v
            meta[k]['_items'] = []
            for v1 in v:
                # note that the following line implys that dicts can
                # only contain classes that extend sifra.structural.Base, or
                # at least have __json_desc__ defined
                nextMeta = deepcopy(class_getter(v1['class']).__json_desc__)
                meta[k]['_items'].append(nextMeta)
                _merge_data_and_metadata(nextMeta, v1)

        else:
            _merge_data_and_metadata(meta[k], data[k])



class Info(str):
    """
    Strings that provide 'metadata' on classes. At present, this is only used to
    identify immutable strings on a class when they are displayed.
    """
    pass



class Pythonizer(object):
    """
    Functor for converting JSONable objects to Python classes.

    Plea

    :param module_name: The name of a Python module.
    """

    def __init__(self, module_name=''):
        self.module_name = module_name


    def __call__(self, obj):
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
                res = class_getter(cls)(**self.__call__(obj))
            else:
                res = {str(k): self.__call__(v) for k, v in obj.iteritems()}
            if attrs is not None:
                res._attributes = attrs
            return res
        if isinstance(obj, list):
            return [self.__call__(v) for v in obj]
        return obj



class Element(object):
    """
    Represents an element of a model. If a model were represented in a relational
    database, this would be analgous to a field in a table.
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
        self.description = Info(description)
        self._default = default
        self.validators = validators

    @property
    def default(self):
        if self._default is False:
            raise NoDefaultException()
        return self._default() if callable(self._default) else self._default

    def __jsonify__(self, val):
        """
        Convert *val* to a form that can be JSON sersialised.
        """

        self.__validate__(val)
        return jsonify(val)

    def __validate__(self, val):
        """
        Validate *val*. This checks that *val* is of subclass ``eval(self.cls)``
        and that no :py:attr:`validators` either return *False* or raise
        exceptions.

        :py:raises:`ValidationError`.
        """

        # Ideally, we'd like to do the following manipulation of self.cls in
        # the constructor. However, at the time the constructor is called, we
        # don't have self.to_python, which is set on this instance in the
        # metaclass StructuralMeta at the time the elements are handled. We
        # could get around this by defining the element class for a module in
        # a way similar to that employed in generate_element_base.
        if isinstance(self.cls, basestring):
            self.cls = [self.to_python.module_name, self.cls]
        try:
            cls = class_getter(self.cls)
            self.cls = [cls.__module__, cls.__name__]
        except:
            # hope that we have a builtin
            cls = eval(self.cls[1])
            self.cls = ['__builtin__', self.cls[1]]

        if not isinstance(val, cls):
            try:
                val = cls(val)
            except:
                raise ValidationError('value is not instance of {}'.format(
                    '.'.join(self.cls)))

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
    """
    Metaclass for structural
    """

    #: Names of :py:class:`Element`s that cannot defined on any class ``c`` for
    #: which ``issubclass(type(c), StructuralMeta)`` is *True*. These are names
    #: of elements which are used internally and for the sake of the performance
    #: of attribute lookup, are banned for other use.
    DISALLOWED_FIELDS = [
        'class',
        'predecessor', '_predecessor',
        '_value',
        '_attributes']

    def __new__(cls, name, bases, dct):
        # check that only one base is instance of _Base
        if len([base for base in bases if issubclass(type(base), StructuralMeta)]) > 1:
            raise MultipleBasesOfTypeBaseError('Invalid bases in class {}'.format(name))

        def extract_params_of_type(clazz):
            # extract the parameters
            params = {}
            for k in dct.keys():
                if isinstance(dct[k], clazz):
                    params[k] = dct.pop(k)

            # cannot have a parameter with name class, as this messes with
            # serialisation
            for field in StructuralMeta.DISALLOWED_FIELDS:
                if field in params:
                    raise DisallowedElementException(
                        'class {} cannot have Element with name "{}"'.format(name, field))

            return params

        dct['__params__'] = extract_params_of_type(Element)

        # create a json description of the class
        json_desc = {}
        for k, v in dct['__params__'].iteritems():
            # TODO: put validators in here
            json_desc[k] = {'class': v.cls}


        for k, v in extract_params_of_type(Info).iteritems():
            json_desc[k] = {
                'class': 'Info',
                'value': str(v)}

        dct['__json_desc__'] = json_desc

        return super(StructuralMeta, cls).__new__(cls, name, bases, dct)

    def __init__(cls, name, bases, dct):
        # we do this here as I prefer to get the module from the class. Not sure
        # if it matters in practice, but it just feels better.
        cls_module = inspect.getmodule(cls).__name__
        cls.to_python = Pythonizer(cls_module)
        cls.__json_desc__['class'] = '.'.join([cls_module, name])

        for param in cls.__params__.itervalues():
            param.to_python = cls.to_python

        for k, v in cls.__json_desc__.iteritems():
            if k == 'class':
                continue
            try:
                ecls = class_getter([cls_module, v['class']])
                if hasattr(ecls, '__json_desc__'):
                    cls.__json_desc__[k] = ecls.__json_desc__
                else:
                    v['class'] = '.'.join([ecls.__module__, ecls.__name__])
                    if isinstance(ecls, Element):
                        try:
                            default = v.default
                        except NoDefaultException:
                            pass
                        else:
                            default = jsonify(default)
                            if default:
                                cls.__json_desc__[k]['default'] = default
            except:
                v['class'] = '.'.join(['__builtin__', v['class']])

        super(StructuralMeta, cls).__init__(name, bases, dct)



class Base(object):
    """
    Base class for all 'model' classes. **This should never be used by clients**
    and serves as a base class for dynamically generated classes returned by
    :py:func:`generate_element_base`, which are designed for use by clients.
    """

    __metaclass__ = StructuralMeta

    def __init__(self, **kwargs):
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

    @property
    def predecessor(self):
        """
        The objects predecessor. This goes back to the database if required.
        """

        if isinstance(self._predecessor, list):
            self._predecessor = self.to_python(self.get_db().get(self._predecessor[0]))
        return self._predecessor

    @classmethod
    def get_db(cls):
        """
        Get the db to be used by this instance.

        .. todo:: This gets the provider from the instance, which allows the
            provider to differ between instances. Not sure if this is desirable
            or not. It may be better to get the provider from the class and,
            perhaps, even make the provider immutable.
        """

        return cls._provider.get_db()

    def __validate__(self):
        """
        Validate this instance.
        """

        pass

    def __jsonify__(self):
        """
        Validate this instance and transform it into an object suitable for
        JSON serialisation.
        """

        self.__validate__()
        res = {'class': [type(self).__module__, type(self).__name__]}
        res.update({
            jsonify(k): v.__jsonify__(getattr(self, k))
            for k, v in self.__params__.iteritems()
            if hasattr(self, k)})
        return res

    def jsonify_with_metadata(self):
        """
        .. note:: It is pretty inefficient duplicating the data as much as we do in
            this method. It would be possible to do this based on the value alone
            (getting the json_desc as required) in the browser. The only trick is,
            being able to differentiate between a list and a dict in javascript.
        """

        meta = deepcopy(self.__json_desc__)
        data = jsonify(self)
        _merge_data_and_metadata(meta, data)
        return meta

    def save(self, category=None, attributes=None):
        """
        Save this instance.
        """

        res = jsonify(self)

        # then we have added something to this.
        if self._predecessor is not None:
            res['predecessor'] = self._predecessor

        self._predecessor = self.get_db().save(
            res,
            category=category,
            attributes=attributes)

        return self._predecessor

    @classmethod
    def load(cls, object_id):
        """
        Load a previously saved instance.
        """

        return cls.to_python(cls._provider.get_db().get(object_id))

    @classmethod
    def set_provider(cls, provider):
        cls._provider = provider



if USE_COUCH_DB:
    from sifra.modelling.serialisation import CouchSerialisationProvider
    Base.set_provider(CouchSerialisationProvider(COUCH_DB_URL, COUCH_DB_NAME))
else:
    from sifra.modelling.serialisation import SqliteSerialisationProvider
    Base.set_provider(SqliteSerialisationProvider())

