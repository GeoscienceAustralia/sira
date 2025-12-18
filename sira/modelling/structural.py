import inspect

from sira.tools.utils import class_getter

# from future.builtins import str, object
# from future.utils import with_metaclass


class NoDefaultException(Exception):
    """
    Thrown when a :py:class:`_Base` is created without providing values for one
    or more :py:class:`Element`s which do not have default values.

    Note that users should never be instantiating or subclassing
    :py:class:`_Base` directly. One should extend a class returned by
    :py:func:`generate_element_base`, which returns a class which extends
    :py:class:`_Base`.
    """


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


# class AlreadySavedException(Exception):
#     """
#     Raised if an attempt is made to save a 'Document' which has previously been
#     saved.
#     """
#     pass


class DisallowedElementException(ValueError):
    """
    Raised if an an attempt is made to define an element with a disallowed
    name. Disallowed names are specified by
    :py:attr:StructuralMeta.DISALLOWED_FIELDS.
    """


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


class Info(str):
    """
    Strings that provide 'metadata' on classes. At present, this is only used to
    identify immutable strings on a class when they are displayed.
    """


class Element(object):
    """
    Represents an element of a model. If a model were represented in a relational
    database, this would be analogous to a field in a table.
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


class StructuralMeta(type):
    """
    Metaclass for structural

    Names of :py:class:`Element`s that cannot be defined on any class ``c`` for
    which ``issubclass(type(c), StructuralMeta)`` is *True*. These are names
    of elements which are used internally and for the sake of the performance
    of attribute lookup, are banned for other use.
    """
    DISALLOWED_FIELDS = [
        'class',
        'predecessor', '_predecessor', '_id',
        '_value',
        '_attributes']

    def __new__(mcs, name, bases, dct):
        # check that only one base is instance of _Base
        if len([base for base in bases if issubclass(type(base), StructuralMeta)]) > 1:
            raise MultipleBasesOfTypeBaseError('Invalid bases in class {}'.format(name))

        def extract_params_of_type(clazz):
            # extract the parameters
            params = {}
            for k in list(dct.keys()):
                if isinstance(dct[k], clazz):
                    params[k] = dct.pop(k)

            # cannot have a parameter with name class, as this messes with
            # serialisation
            for field in StructuralMeta.DISALLOWED_FIELDS:
                if field in params:
                    raise DisallowedElementException(
                        'class {} cannot have Element with name "{}"'.
                        format(name, field))

            return params

        dct['__params__'] = extract_params_of_type(Element)

        # create a json description of the class
        json_desc = {}
        for k, v in list(dct['__params__'].items()):
            # Future improvement: add validators here.  # noqa: W0511
            json_desc[k] = {'class': v.cls}

        for k, v in list(extract_params_of_type(Info).items()):
            json_desc[k] = {
                'class': 'Info',
                'value': str(v)}

        dct['__json_desc__'] = json_desc

        return super(StructuralMeta, mcs).__new__(mcs, name, bases, dct)

    def __init__(cls, name, bases, dct):
        # We do this here as I prefer to get the module from the class. Not sure
        # if it matters in practice, but it feels better. cls_module contains
        # the module in which this class is defined and we know that the types
        # declared for the Elements of a class are accessible in that module.
        cls_module = inspect.getmodule(cls).__name__
        cls.__json_desc__['class'] = '.'.join([cls_module, name])

        for param in list(cls.__params__.values()):
            param.cls_module = cls_module

        for k, v in list(cls.__json_desc__.items()):
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
                            # default = jsonify(default)
                            if default:
                                cls.__json_desc__[k]['default'] = default
            except Exception:  # pylint: disable=broad-except
                v['class'] = '.'.join(['__builtin__', v['class']])

        super(StructuralMeta, cls).__init__(name, bases, dct)


class Base(metaclass=StructuralMeta):
    """
    Base class for all 'model' classes. **This should never be used by clients**
    and serves as a base class for dynamically generated classes returned by
    :py:func:``, which are designed for use by clients.
    """

    def __init__(self, **kwargs):
        self._predecessor = kwargs.pop('predecessor', None)

        if self._predecessor is None:
            # then we provide default values for each element
            for k, v in list(self.__params__.items()):  # pylint: disable=no-member
                if k not in kwargs:
                    try:
                        kwargs[k] = v.default
                    except NoDefaultException as error:
                        raise ValueError('Must provide value for {}'.format(k)) from error
        for k, v in list(kwargs.items()):
            setattr(self, k, v)
