import numpy as np
import os
import csv

class Hazard(object):
    """
    handel a singloe
    """

    def __init__(self,hazard_scenario_name, scenario_hazard_data):
        self.hazard_scenario_name = hazard_scenario_name
        self.scenario_hazard_data = scenario_hazard_data
        self.round_off = 2

    def get_hazard_intensity_at_location(self, longitude, latitude):

        for comp in self.scenario_hazard_data:

            if round(float(comp["longitude"]), self.round_off) == round(float(longitude), self.round_off):
                if round(float(comp["latitude"]), self.round_off) == round(float(latitude), self.round_off):
                    return comp["hazard_intensity"]

        raise Exception("Invalid Values for Longitude or Latitude")

    def get_seed(self):
        seed = 0
        for i, letter in enumerate(self.hazard_scenario_name):
            seed = seed + (i + 1) * ord(letter)
        return seed

    def __str__(self):

        output = self.hazard_scenario_name+'\n'
        for hazrd in self.scenario_hazard_data:
            output = output + "longitude: "+str(hazrd["longitude"]) + " latitude: " + str(hazrd["latitude"]) +" hazard_intensity: "+ str(hazrd["hazard_intensity"]) +'\n'
        return output

class HazardsContainer(object):
    """
    The idea is to abstract the number and type of hazards to allow greater
    flexibility in the type and number of hazards to be modelled.
    """
    def __init__(self, configuration):

        # string variables
        self.listOfhazards = []
        self.hazard_type = configuration.HAZARD_TYPE
        self.intensity_measure_param = configuration.INTENSITY_MEASURE_PARAM
        self.intensity_measure_unit = configuration.INTENSITY_MEASURE_UNIT

        # get hazard data from scenario file
        if configuration.HAZARD_INPUT_METHOD == "scenario_file":
            self.scenario_hazard_data, self.hazard_scenario_list = \
                HazardsContainer.populate_scenario_hazard_data_using_hazard_file(configuration.SCENARIO_FILE)

            self.num_hazard_pts = len(self.hazard_scenario_list)

        # get hazard data from an array of hazard intensity values
        elif configuration.HAZARD_INPUT_METHOD == "hazard_array":

            self.num_hazard_pts = \
                int(round((configuration.INTENSITY_MEASURE_MAX - configuration.INTENSITY_MEASURE_MIN) /
                          float(configuration.INTENSITY_MEASURE_STEP) + 1))

            # using the limits and step generate a list of hazard intensity values
            self.hazard_scenario_list = np.linspace(configuration.INTENSITY_MEASURE_MIN,
                                                    configuration.INTENSITY_MEASURE_MAX,
                                                    num=self.num_hazard_pts)

            # containing hazard value for each location
            self.scenario_hazard_data, self.hazard_scenario_list = \
                HazardsContainer.populate_scenario_hazard_data_using_hazard_array(self.hazard_scenario_list)

        for hazard_scenario_name in self.scenario_hazard_data.keys():
            self.listOfhazards.append(Hazard(hazard_scenario_name, self.scenario_hazard_data[hazard_scenario_name]))

        self.hazard_scenario_name = self.hazard_scenario_list

    @staticmethod
    def populate_scenario_hazard_data_using_hazard_file(scenario_file):
        root = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(root, "hazard", scenario_file )
        scenario_hazard_data = {}

        with open(csv_path, "rb") as f_obj:
            reader = csv.DictReader(f_obj, delimiter=',')

            hazard_scenario_list = [scenario for scenario in reader.fieldnames if
                                    scenario not in ["longitude", "latitude"]]

            for scenario in hazard_scenario_list:
                scenario_hazard_data[scenario] = []

            for row in reader:
                for col in row:
                    if col not in ["longitude", "latitude"]:
                        hazard_intensity = row[col]
                        scenario_hazard_data[col].append(
                            {"longitude": row["longitude"], "latitude": row["latitude"],
                             "hazard_intensity": hazard_intensity})

        return scenario_hazard_data, hazard_scenario_list

    @staticmethod
    def populate_scenario_hazard_data_using_hazard_array(num_hazard_pts):

        scenario_hazard_data = {}
        hazard_scenario_list = []
        for i, hazard_intensity in enumerate(num_hazard_pts):
            hazard_scenario_list.append("s_"+str(i))
            scenario_hazard_data["s_"+str(i)] = [{"longitude": 0, "latitude": 0, "hazard_intensity": hazard_intensity}]

        return scenario_hazard_data, hazard_scenario_list


