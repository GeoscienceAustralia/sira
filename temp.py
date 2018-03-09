import json
from sifra.configuration import Configuration
import pandas as pd
import os

from sifra.configuration import Configuration
from sifra.scenario import Scenario

from sifra.modelling.system_topology import SystemTopology
from sifra.infrastructure_response import calculate_response, post_processing
from sifra.logger import rootLogger


jsonFileName = "C:\\Users\\u12089\\Desktop\\sifra\\config_test.json"

config = Configuration(jsonFileName)
print(config.SYS_CONF_FILE)
with open(config.SYS_CONF_FILE, 'r') as f:
    model = json.load(f)

component_list = model['component_list']
node_conn_df = model['node_conn_df']
sysinp_setup = model['sysinp_setup']
sysout_setup = model['sysout_setup']
fragility_data = model['fragility_data']


print('*******************************JSON***************************************************')

# fragility_data
damage_state_df = model['damage_state_df']

# for var in damage_state_df:
#     print(var)
#     component_type_and_damage_state_pair = eval(var)
#     print(component_type_and_damage_state_pair)
#     break

for key in fragility_data:
    response_params = {}

    if key not in damage_state_df:
        response_params["damage_state_description"] = u"NA"
    else:
        response_params["damage_state_description"] = damage_state_df[key][u'damage_state_definitions']

    print(response_params["damage_state_description"])

print('*******************************XLSX***************************************************')

xlsxFileName = config.SYS_CONF_FILE.split('.')[0]+'.xlsx'


comp_type_dmg_algo = pd.read_excel(
    xlsxFileName, sheet_name='comp_type_dmg_algo',
    index_col=[0, 1], header=0,
    skiprows=0, skipinitialspace=True)

from sifra.model_ingest import ingest_model

# run system
# config.SYS_CONF_FILE=xlsxFileName
# scenario=Scenario(config)
# infrastructure, algorithm_factory = ingest_model(config)
# scenario.algorithm_factory = algorithm_factory
# sys_topology_view = SystemTopology(infrastructure, scenario)
# sys_topology_view.draw_sys_topology(viewcontext="as-built")
# post_processing_list = calculate_response(scenario, infrastructure)
# post_processing(infrastructure, scenario, post_processing_list)
