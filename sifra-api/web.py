import json
from functools import wraps

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect

from sifra.modelling.component_db import getComponentCategories, getAllInstances, getInstance
from sifra.modelling.responsemodels import *
from sifra.modelling.structural import class_getter
from sifra.modelling.utils import get_all_subclasses, pythonify

csrf = CSRFProtect()
app = Flask(__name__)
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'super-secret'
CORS(app)


class InvalidUsage(Exception):
    status_code = 500

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


def _errorWrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception, e:
            raise InvalidUsage(str(e))
    return wrapper


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/')
@_errorWrapper
def index():
    return jsonify({'version': 0.0, 'message': 'Hello!'})


@app.route('/sector-lists')
@_errorWrapper
def componentTopology():
    # expecting args with one or more of {hazard, sector, facility_type, component}
    #print '\n\ngetting components\n\n', getComponentCategories(
    #    **{k.strip(): v.strip() for k, v in request.args.iteritems()})
    return jsonify(getComponentCategories(
        **{k.strip(): v.strip() for k, v in request.args.iteritems()}))


@app.route('/class-types')
@_errorWrapper
def classTypes():
    # at the moment this just returns response models
    res = [c.__json_desc__['class'] for c in ResponseModel.__subclasses__()]
    return jsonify(res)


@app.route('/class-def/<class_name>')
@_errorWrapper
def classDef(class_name):
    for clazz in ResponseModel.__subclasses__():
        if clazz.__json_desc__['class'] == class_name:
            return jsonify(clazz.__json_desc__)
    return jsonify({'error': 'class "{}" does not exist'.format(class_name)})


@app.route('/sub-classes-of/<class_name>')
@_errorWrapper
def subClassesOf(class_name):
    return jsonify(get_all_subclasses(class_getter(class_name.rsplit('.', 1))))


@app.route('/instances-of/<class_name>')
@_errorWrapper
def instancesOf(class_name):
    return jsonify(getAllInstances(class_getter(class_name.rsplit('.', 1))))


@app.route('/instance/<instance_name>')
@_errorWrapper
def instance(instance_name):
    inst = pythonify(getInstance(instance_name))
    return jsonify(inst.jsonify_with_metadata())


@csrf.exempt
@app.route('/save', methods=['POST'])
@_errorWrapper
def save():
    data = json.loads(request.data)
    category = data.pop('component_sector', None)
    attrs = data.pop('attributes', None)
    inst = pythonify(data)
    oid = inst.save(
        category=category,
        attributes=attrs)
    name = attrs.pop('name', 'anonymous component') if attrs is not None else 'anonymous component'
    return jsonify({'id': oid, 'name': name})
