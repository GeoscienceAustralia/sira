import openpyxl

# read excel file
# remove the first three lines from the 5 requrired sheets
# create a news sheet with name and format as described in the one

def main():
    fileName='C:\\Users\\u12089\\Desktop\\sifra\\models\potable_water_treatment_plant\\sysconfig_pwtp_400ML_delete.xlsx'
    book = openpyxl.load_workbook(fileName)

    required_sheets = ['component_list', 'component_connections', 'supply_setup', 'output_setup',
                            'comp_type_dmg_algo', 'damage_state_def']
    for sheet in book.sheetnames:
        if sheet in required_sheets:
            print(sheet)



if __name__ == "__main__":
    main()
