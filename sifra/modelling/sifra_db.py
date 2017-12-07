from sqlalchemy import Integer, String, create_engine, Column, ForeignKey, and_
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.declarative import declarative_base
from sifra.settings import DB_CONNECTION_STRING, BUILD_DB_TABLES
from sifra.modelling.utils import get_all_subclasses

from sifra.modelling.utils import jsonify, pythonify

import json

_Base = declarative_base()


class ModelTable(_Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True)
    name = Column(String(), unique=True)
    clazz = Column(String())

    components_json = Column(String())
    attributes = relationship('ModelAttribute',
                              primaryjoin="ModelTable.id==ModelAttribute.model_id",
                              cascade='all, delete-orphan')


class ModelAttribute(_Base):
    __tablename__ = "model_attributes"
    model_id = Column(Integer, ForeignKey('models.id'), primary_key=True)
    name = Column(String(100), primary_key=True)
    value = Column(String())


class ComponentTable(_Base):
    __tablename__ = "components"
    id = Column(Integer, primary_key=True)
    component_type = Column(String(), unique=True)
    clazz = Column(String())

    component_json = Column(String())
    attributes = relationship('ComponentAttribute',
                              primaryjoin="ComponentTable.id==ComponentAttribute.component_id",
                              cascade='all, delete-orphan')

class ComponentAttribute(_Base):
    __tablename__ = "component_attributes"
    component_id = Column(Integer, ForeignKey('components.id'), primary_key=True)
    name = Column(String(100), primary_key=True)
    value = Column(String())


def save_model(model):
    res = jsonify(model)

    clazz = '.'.join(res['class'])
    model_name = res['name']
    session = _Session()

    # Extract the attributes of the jsonified class
    model_attributes = dict()
    # model_attributes['type'] = 'model'
    for var_name in res.keys():
        if var_name in model.__dict__ and is_attribute_type(getattr(model, var_name)):
            model_attributes[var_name] = res.pop(var_name)

    try:
        model_db = ModelTable(name=model_name,
                            clazz=clazz,
                            components_json=json.dumps(res))

        session.add(model_db)
        session.commit()

        # _addAttributes(res)
        for m_name, m_value in model_attributes.items():
            model_attribute = ModelAttribute(model_id=model_db.id,
                                             name=m_name,
                                             value=m_value)
            session.add(model_attribute)

        session.commit()

        return model_db.id
    except Exception, e:
        session.rollback()
        raise e
    finally:
        session.close()


def load_model(obj_id):
    session = _Session()
    model_row = session.query(ModelTable).filter(ModelTable.id == obj_id).one()
    model_kwargs = json.loads(model_row.components_json)
    for ma in model_row.attributes:
        model_kwargs[ma.name] = ma.value

    res = pythonify(model_kwargs)
    session.close()
    return res


def is_attribute_type(var):
    return isinstance(var, (basestring, int, float))


def save_component(component):
    res = jsonify(component)

    clazz = '.'.join(res['class'])
    component_type = res['component_type']
    session = _Session()

    # Extract the attributes of the jsonified class
    comp_attributes = dict()
    # comp_attributes['type'] = 'component'
    for var_name in res.keys():
        if var_name in component.__dict__ and is_attribute_type(getattr(component, var_name)):
            comp_attributes[var_name] = res.pop(var_name)

    try:
        comp_row = ComponentTable(component_type=component_type,
                                  clazz=clazz,
                                  component_json=json.dumps(res))

        session.add(comp_row)
        session.commit()

        # _addAttributes(res)
        for m_name, m_value in comp_attributes.items():
            comp_attribute = ComponentAttribute(component_id=comp_row.id,
                                                name=m_name,
                                                value=m_value)
            session.add(comp_attribute)

        session.commit()

        return comp_row.id
    except Exception, e:
        session.rollback()
        raise e
    finally:
        session.close()


def getAllInstances(cls):
    session = _Session()
    components = session.query(ComponentTable).filter(
        ComponentTable.clazz.in_(get_all_subclasses(cls)))
    return [{'name': comp.component_type, 'id': comp.id} for comp in components.all()]


def getComponentCategories(hazard=None, sector=None, facility_type=None, component=None):
    # convert anything that evaluates to false to None
    hazard = hazard or None
    sector = sector or None
    facility_type = facility_type or None
    component = component or None

    session = _Session()
    rows = session.query(ComponentAttribute)

    response = dict()

    rows = session.query(ComponentAttribute)
    if hazard is not None:
        rows = rows.filter(and_(ComponentAttribute.value == hazard,
                                ComponentAttribute.name == 'hazard_type'))
        response.update({'hazards': list(set((row.value for row in rows)))})
    else:
        rows = rows.filter(ComponentAttribute.name == 'hazard_type')
        response.update({'hazards': list(set((row.value for row in rows)))})

    rows = session.query(ComponentAttribute)
    if sector is not None:
        rows = rows.filter(and_(ComponentAttribute.value == sector,
                                ComponentAttribute.name == 'component_class'))
        response.update({'sectors': list(set((row.value for row in rows)))})
    else:
        rows = rows.filter(ComponentAttribute.name == 'component_class')
        response.update({'sectors': list(set((row.value for row in rows)))})

    rows = session.query(ComponentAttribute)
    if facility_type is not None:
        rows = rows.filter(ComponentAttribute.facility_type == facility_type)
        response.update({'facilities': list(['Powerstation', 'Water treatment plant'])})
    else:
        response.update({'facilities': list(['Powerstation', 'Water treatment plant'])})

    rows = session.query(ComponentAttribute)
    if component is not None:
        rows = rows.filter(and_(ComponentAttribute.value == component,
                                ComponentAttribute.name == 'component_type'))
        response.update({'components': list(set((row.value for row in rows)))})
    else:
        rows = rows.filter(ComponentAttribute.name == 'component_type')
        response.update({'components': list(set((row.value for row in rows)))})

    return response




def getInstance(instance_id):
    session = _Session()
    return json.loads(session.query(ComponentTable).filter(
        ComponentTable.id == instance_id).one().component_json)


_engine = create_engine(
    DB_CONNECTION_STRING,
    echo=False,
    connect_args={'check_same_thread': False},
    poolclass=StaticPool)
_Session = sessionmaker(bind=_engine)


if BUILD_DB_TABLES:
    _Base.metadata.create_all(_engine)

