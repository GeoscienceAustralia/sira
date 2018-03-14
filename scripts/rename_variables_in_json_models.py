import os

python_file_paths = []
for root, dir_names, file_names in os.walk(os.path.dirname(os.getcwd())):
    for file_name in file_names:
        if file_name.endswith('.json'):
            if 'models' in root:
                python_file_paths.append(os.path.join(root, file_name))


for python_file_path in python_file_paths:
    data = ''
    with open(python_file_path, 'r') as input_file:
        data = input_file.read()
        data = data.replace('damage_median', 'median')
        data = data.replace('damage_logstd', 'beta')

    with open(python_file_path, "w") as output_file:
        output_file.write(data)
