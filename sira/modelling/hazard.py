import csv
import json
from collections import OrderedDict
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd


def haversine_dist(lon1, lat1, lon2, lat2):
    """
    Calculates the great circle distance between two points
    on the earth (specified in decimal degrees)
    Based on post by: Michael Dunn
    https://stackoverflow.com/a/4913653
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometres. For miles, use 3956.
    earth_raduis_km = 6372.8

    return c * earth_raduis_km


def find_nearest(lon, lat, df, col):
    """
    Matches elements based on nearest Haversine distance,
    given:
        a set of longitude and latitude values,
        a dataframe,
        a column header to return
    returns:
        the value in the matching index and provided column
    """
    distances = df.apply(
        lambda row: haversine_dist(lon, lat, row['pos_x'], row['pos_y']),
        axis=1
    )
    min_idx = distances.idxmin()
    return df.loc[min_idx, col]


class Hazard(object):

    def __init__(self,
                 hazard_scenario_name,
                 scenario_hazard_data,
                 hazard_input_method,
                 hazard_data_df):
        self.hazard_scenario_name = hazard_scenario_name
        self.scenario_hazard_data = scenario_hazard_data
        self.hazard_input_method = hazard_input_method
        self.hazard_data_df = hazard_data_df
        self.round_off = 2

    def get_hazard_intensity(self, x_location, y_location):
        if str(self.hazard_input_method).lower() == 'calculated_array':
            for comp in self.scenario_hazard_data:
                return comp["hazard_intensity"]
        if str(self.hazard_input_method).lower() == 'hazard_file':
            hazard_intensity = find_nearest(
                x_location, y_location,
                self.hazard_data_df, self.hazard_scenario_name
            )
            return hazard_intensity

        raise Exception("Invalid values for component location.")

    def get_seed(self):
        seed = 0
        for i, letter in enumerate(self.hazard_scenario_name):
            seed = seed + (i + 1) * ord(letter)
        return seed

    def __str__(self):
        output = self.hazard_scenario_name + '\n'
        for hazrd in self.scenario_hazard_data:
            output = output\
                + "pos_x: " + str(hazrd["pos_x"])\
                + " pos_y: " + str(hazrd["pos_y"])\
                + " hazard_intensity: " + str(hazrd["hazard_intensity"])\
                + '\n'
        return output


class HazardsContainer(object):
    """
    The idea is to abstract the number and type of hazards to allow greater
    flexibility in the type and number of hazards to be modelled.
    """
    def __init__(self, configuration, model_file_path):

        # string variables
        self.listOfhazards = []
        self.hazard_type = configuration.HAZARD_TYPE
        self.intensity_measure_param = configuration.HAZARD_INTENSITY_MEASURE_PARAM
        self.intensity_measure_unit = configuration.HAZARD_INTENSITY_MEASURE_UNIT
        self.focal_hazard_scenarios = configuration.FOCAL_HAZARD_SCENARIOS

        # ---------------------------------------------------------------------
        # Read in the model JSON file into a dict:
        with open(model_file_path, 'r') as jfile:
            ci_model = json.load(jfile)

        # List of System Components with their attributes:
        component_table = pd.DataFrame.from_dict(
            ci_model['component_list'], orient="index")

        # Prepare a dataframe to hold hazard attribution for each component location:
        self.component_loc = component_table[['pos_x', 'pos_y']].copy()

        # ---------------------------------------------------------------------
        # Get hazard data from scenario file
        if configuration.HAZARD_INPUT_METHOD.lower() == "hazard_file":
            hazard_file_path = configuration.HAZARD_INPUT_FILE
            self.hazard_table = pd.read_csv(hazard_file_path)
            self.scenario_hazard_data, self.hazard_scenario_list = \
                HazardsContainer.populate_scenario_hazard_using_hazard_file(
                    hazard_file_path)

            self.num_hazard_pts = len(self.hazard_scenario_list)

        # ---------------------------------------------------------------------
        # Set up hazard data from an array of hazard intensity values
        elif configuration.HAZARD_INPUT_METHOD.lower() == "calculated_array":
            self.hazard_table = None
            self.num_hazard_pts = \
                int(round((configuration.INTENSITY_MEASURE_MAX
                           - configuration.INTENSITY_MEASURE_MIN) /
                          float(configuration.INTENSITY_MEASURE_STEP) + 1
                          )
                    )

            # Using the limits and step generate a list of hazard
            # intensity values
            self.hazard_scenario_list \
                = np.linspace(configuration.INTENSITY_MEASURE_MIN,
                              configuration.INTENSITY_MEASURE_MAX,
                              num=self.num_hazard_pts)

            # containing hazard value for each location
            self.scenario_hazard_data, self.hazard_scenario_name = \
                HazardsContainer.populate_scenario_hazard_using_hazard_array(
                    self.hazard_scenario_list)
            self.hazard_scenario_list = ["%0.3f" % np.float(x)
                                         for x in self.hazard_scenario_list]

        # ---------------------------------------------------------------------
        for hazard_scenario_name in self.scenario_hazard_data.keys():
            self.listOfhazards.append(
                Hazard(
                    hazard_scenario_name,
                    self.scenario_hazard_data[hazard_scenario_name],
                    configuration.HAZARD_INPUT_METHOD,
                    self.hazard_table
                )
            )

    def get_listOfhazards(self):
        for hazard_intensity in self.listOfhazards:
            yield hazard_intensity

    @staticmethod
    def populate_scenario_hazard_using_hazard_file(hazard_file_path):
        """
        :hazard_file_path: this should be the full or relative path to the
        hazard specification file
        This file must have "pos_x" and "pos_y" as the first two headers.
        These are equivalent to longitude and latitude respectively.
        """
        scenario_hazard_data = {}
        hazard_input = pd.read_csv(hazard_file_path)
        hazard_scenario_list = hazard_input.columns.to_list()[2:]

        with open(hazard_file_path, "r") as f_obj:
            reader = csv.DictReader(f_obj, delimiter=',')

            for scenario in hazard_scenario_list:
                scenario_hazard_data[scenario] = []

            for row in reader:
                for col in row:
                    if col not in ["pos_x", "pos_y"]:
                        hazard_intensity = row[col]
                        scenario_hazard_data[col].append(
                            {"pos_x": row["pos_x"],
                             "pos_y": row["pos_y"],
                             "hazard_intensity": hazard_intensity})

        return scenario_hazard_data, hazard_scenario_list

    @staticmethod
    def populate_scenario_hazard_using_hazard_array(num_hazard_pts):

        scenario_hazard_data = OrderedDict()
        hazard_scenario_name = []
        for i, hazard_intensity in enumerate(num_hazard_pts):
            hazard_scenario_name.append("s_" + str(i))
            scenario_hazard_data["s_" + str(i)] \
                = [{"pos_x": 0,
                    "pos_y": 0,
                    "hazard_intensity": hazard_intensity}]

        return scenario_hazard_data, hazard_scenario_name
