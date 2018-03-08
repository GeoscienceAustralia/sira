import json
from sifra.configuration import Configuration
import pandas as pd

jsonFileName="C:\\Users\\u12089\\Desktop\\sifra\\config_test.json"
config = Configuration(jsonFileName)
with open(config.SYS_CONF_FILE, 'r') as f:
    model = json.load(f)

component_list = model['component_list']
node_conn_df = model['node_conn_df']
sysinp_setup = model['sysinp_setup']
sysout_setup = model['sysout_setup']
fragility_data = model['fragility_data']




print('*******************************JSON***************************************************')


damage_state_df = model['damage_state_df']

for var in damage_state_df:
    # index = (unicode(eval(var)[0]), unicode(eval(var)[1]))
    # print(damage_state_df[var])
    # print(pd.to_json(damage_state_df[var]))
    break

print('*******************************XLSX***************************************************')

xlsxFileName = config.SYS_CONF_FILE.split('.')[0]+'.xlsx'
damage_state_def = pd.read_excel(
    xlsxFileName, sheet_name='damage_state_def',
    index_col=[0, 1], header=0,
    skiprows=0, skipinitialspace=True)

damage_def_dict={}
for index, damage_def in damage_state_def.iterrows():
    # print(damage_def.values)

    # print(damage_def.values)
    # damage_def_dict[index] = damage_def
    break

from sifra.model_ingest import ingest_model


config.SYS_CONF_FILE=xlsxFileName
ingest_model(config)