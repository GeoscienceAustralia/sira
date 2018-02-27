# class Infrastructure:
#     def __init__(self, facility_definition_file):
#         #read

import  xlrd
filePath = "C:\Users\u12089\Desktop\sifra-dev\models\potable_water_treatment_plant\sysconfig_pwtp_400ML.xlsx"

wb = xlrd.open_workbook(filePath)


for sheet_name in wb.sheet_names():
    sheet = wb.sheet_by_name(sheet_name)

    print(sheet.cell(0, 0))

    cells = sheet.row_slice(rowx=0,
                                  start_colx=0,
                                  end_colx=2)
    for cell in cells:
        print cell.value
