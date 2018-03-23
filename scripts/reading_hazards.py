import csv

class Component:
    lat = 0
    lng = 0
    hzd = 0

standard_format = {}

csv_path = "C:\\Users\\u12089\\Desktop\\sifra-dev\\hazard\\data.csv"


with open(csv_path, "rb") as f_obj:
    reader = csv.DictReader(f_obj, delimiter=',')
    for line in reader:
        print(line["longitude"],line["latitude"])
