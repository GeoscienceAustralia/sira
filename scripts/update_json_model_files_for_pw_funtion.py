import os

python_file_paths = []
for root, dir_names, file_names in os.walk(os.path.dirname(os.getcwd())):
    for file_name in file_names:
        if file_name.endswith('.json'):
            if 'models' in root:
                if file_name is not 'sysconfig_simple_parallel.json':
                    python_file_paths.append(os.path.join(root, file_name))


#merge two sheets
# convert faigility data objects into standard objects!!
print(python_file_paths)


# move "damage_ratio": 0.3, and     "functionality": 1, as properties in damage_functions
'''
"['SYSTEM_OUTPUT','DS1 Slight']": {
      "functionality": 1,
      "damage_ratio": 0.01,
      
      "damage_functions": [
        {
          "damage_function_name": "LogNormalCDF",
          "beta": 0.01,
          "median": 10.0,
          "fragility_source": null,
          "minimum": null,
          "damage_state_definition": "Not Available."
        }
      ],
      "recovery_parameters": {
        "recovery_95percentile": 0.1,
        "recovery_mean": 1.0,
        "recovery_std": 0.0607956832
      }
    },
'''

# delete "mode": 1,  "sigma_1": null,    "sigma_2": null,