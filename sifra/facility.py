# class Facility:
#     def __init__(self, facility_definition_file):
#         #read

import xlrd

#create varable that hold excel data
#use this class to pass data to the required classes

# TODO figure out exactly in which formate the data is converted into and convet to exactly that format
# TODO class should hold funtion to initally read excel data -> convert it into the required format
# TODO then create functions for json to read data exactly like it needs to be submitted forward
# Since nither of the sheets are realated to each other there is no point in having one big excel file
# create smaller seperate json file for each task smaler the file the faster its read and written
# figure out how the data for each sheet is flowing its starting in ingest_model and constructor of IFFACULTY
component_connections
component_connections
output_setup
supply_setup
comp_type_dmg_algo
damage_state_def

filePath = "C:\Users\u12089\Desktop\sifra-dev\models\potable_water_treatment_plant\sysconfig_pwtp_400ML.xlsx"
sheetNames = ["component_connections", "component_connections", "output_setup", "supply_setup", "comp_type_dmg_algo", "damage_state_def"]

wb = xlrd.open_workbook(filePath)


for sheet_name in wb.sheet_names():
    sheet = wb.sheet_by_name(sheet_name)

    print(sheet.cell(0, 0))

    cells = sheet.row_slice(rowx=0,
                                  start_colx=0,
                                  end_colx=2)
    for cell in cells:
        print cell.value
