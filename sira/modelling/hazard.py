import csv
import bisect
import json
from collections import OrderedDict
from math import asin, cos, radians, sin, sqrt
import numpy as np
import pandas as pd
from pathlib import Path
import logging
rootLogger = logging.getLogger(__name__)


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
    earth_radius_km = 6372.8

    return c * earth_radius_km


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

    def __init__(
            self,
            hazard_event_id,
            hazard_input_method,
            hazard_intensity_header,
            hazard_data_df
    ):
        self.hazard_event_id = hazard_event_id
        self.hazard_input_method = hazard_input_method
        self.hazard_data_df = hazard_data_df
        self.hazard_intensity_header = hazard_intensity_header
        # self.hazard_data_dict = hazard_data_dict
        self.round_off = 3
        self.hazard_values_applied = []

    def get_hazard_intensity(self, x_location, y_location, site_id=None):
        if str(self.hazard_input_method).lower() == 'calculated_array':
            event_id = str(self.hazard_event_id)
            site_id = '0'
            return self.hazard_data_df.loc[event_id, site_id]
            # for haz in self.hazard_data_dict:
            #     return haz[self.hazard_intensity_header]
        if str(self.hazard_input_method).lower() in ['hazard_file', 'scenario_file']:
            # -----------------------------------------------------------------------
            if site_id is None:
                hazard_intensity = find_nearest(
                    x_location, y_location,
                    self.hazard_data_df, self.hazard_event_id)
            # -----------------------------------------------------------------------
            # To ignore non-costed nodes, their site_id is given as a negative value
            elif float(site_id) < 0:
                hazard_intensity = 0
            # -----------------------------------------------------------------------
            else:
                event_id = str(self.hazard_event_id)
                site_id = str(site_id)
                try:
                    hazard_intensity = self.hazard_data_df.loc[event_id, site_id]
            # -----------------------------------------------------------------------
                except KeyError:
                    print(f"No hazard value for: event_id {event_id}, site_id {site_id}")
                    hazard_intensity = 0
            # -----------------------------------------------------------------------
            return hazard_intensity
        raise Exception("Invalid values for component location.")

    def get_seed(self):
        seed = 0
        for i, letter in enumerate(self.hazard_event_id):
            seed = seed + (i + 1) * ord(letter)
        return seed

    def __str__(self):
        return {
            "Data shape": self.hazard_data_df.shape,
            "Headers": self.hazard_data_df.columns
        }


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
        self.HAZARD_INPUT_HEADER = configuration.HAZARD_INPUT_HEADER
        self.HAZARD_SCALING_FACTOR = configuration.HAZARD_SCALING_FACTOR

        # ---------------------------------------------------------------------
        # Read in the model JSON file into a dict:
        with open(model_file_path, 'r') as jfile:
            ci_model = json.load(jfile)

        # List of System Components with their attributes:
        component_table = pd.DataFrame.from_dict(
            ci_model['component_list'], orient="index")

        # Prepare a dataframe to hold hazard attribution for each component location:
        if configuration.HAZARD_INPUT_METHOD.lower() in ['hazard_file', 'scenario_file']:
            self.component_loc = component_table[['pos_x', 'pos_y', 'site_id']].copy()
        if configuration.HAZARD_INPUT_METHOD.lower() == "calculated_array":
            self.component_loc = component_table[['pos_x', 'pos_y']].copy()

        # ---------------------------------------------------------------------
        # Get hazard data from SCENARIO FILE
        # ---------------------------------------------------------------------
        if configuration.HAZARD_INPUT_METHOD.lower() in ['hazard_file', 'scenario_file']:
            hazard_file_path = configuration.HAZARD_INPUT_FILE
            (
                self.hazard_data_df,
                self.hazard_scenario_list,
                self.hazard_intensity_list
            ) = HazardsContainer.populate_scenario_hazard_using_hazard_file(
                hazard_file_path,
                self.HAZARD_INPUT_HEADER,
                self.HAZARD_SCALING_FACTOR)

            self.num_hazard_pts = len(self.hazard_scenario_list)

            for hazard_event_id in self.hazard_scenario_list:
                self.listOfhazards.append(
                    Hazard(
                        hazard_event_id,
                        configuration.HAZARD_INPUT_METHOD,
                        self.HAZARD_INPUT_HEADER,
                        self.hazard_data_df,
                        # self.hazard_data_dict,
                    )
                )

        # ---------------------------------------------------------------------
        # Set up hazard data from a CALCULATED ARRAY of hazard intensity values
        # ---------------------------------------------------------------------
        elif configuration.HAZARD_INPUT_METHOD.lower() == "calculated_array":
            self.hazard_data_df = None
            self.num_hazard_pts = int(
                round(
                    (configuration.INTENSITY_MEASURE_MAX \
                        - configuration.INTENSITY_MEASURE_MIN) \
                    / float(configuration.INTENSITY_MEASURE_STEP) + 1
                )
            )

            # Using the limits and step generate a list of hazard intensity values
            self.hazard_intensity_list \
                = np.linspace(
                    configuration.INTENSITY_MEASURE_MIN,
                    configuration.INTENSITY_MEASURE_MAX,
                    num=self.num_hazard_pts)

            (self.hazard_data_df, self.hazard_scenario_list) = \
                HazardsContainer.populate_scenario_hazard_using_hazard_array(
                    self.hazard_intensity_list, self.HAZARD_INPUT_HEADER)

            for hazard_event_id in self.hazard_scenario_list:
                self.listOfhazards.append(
                    Hazard(
                        hazard_event_id,
                        configuration.HAZARD_INPUT_METHOD,
                        self.HAZARD_INPUT_HEADER,
                        self.hazard_data_df,
                        # self.hazard_data_dict[hazard_event_id],
                    )
                )
    # -------------------------------------------------------------------------

    def get_listOfhazards(self):
        for hazard_obj in self.listOfhazards:
            yield hazard_obj

    @staticmethod
    def populate_scenario_hazard_using_hazard_file(
            hazard_file_path, hazard_header, SCALING_FACTOR=1.0):
        """
        :hazard_file_path: this should be the full or relative path to the
        hazard specification file
        This file must have "pos_x" and "pos_y" as the first two headers.
        These are equivalent to longitude and latitude respectively.
        """
        event_header = 'event_id'
        site_header = 'site_id'
        dtypes = {
            site_header: 'str',
            event_header: 'str',
            hazard_header: 'float32'
        }
        hazard_file_path = Path(hazard_file_path)
        rootLogger.info("Reading in hazard data file...")
        if hazard_file_path.suffix == '.csv':
            hazard_input_df = pd.read_csv(hazard_file_path, dtype=dtypes)
        if hazard_file_path.suffix == '.parquet':
            hazard_input_df = pd.read_parquet(hazard_file_path)
            hazard_input_df.reset_index(inplace=True)
        rootLogger.info("    Completed ingesting hazard data!")

        hazard_input_df[hazard_header] *= SCALING_FACTOR

        hazard_scenario_list = list(np.unique(hazard_input_df['event_id']))
        hazard_intensity_list = hazard_input_df[hazard_header].values

        hazard_input_df.set_index([event_header, site_header], inplace=True, drop=True)
        hazard_input_df = hazard_input_df.unstack(level=-1, fill_value=0)
        hazard_input_df = hazard_input_df.droplevel(axis='columns', level=0)

        # return hazard_input_df, hazard_data_dict, hazard_scenario_list, hazard_intensity_list
        return hazard_input_df, hazard_scenario_list, hazard_intensity_list

    @staticmethod
    def populate_scenario_hazard_using_hazard_array(hazard_intensity_list, hazard_header):

        hazard_scenario_list = [
            "%0.3f" % float(x) for x in hazard_intensity_list
        ]

        event_header = 'event_id'
        site_header = 'site_id'
        dtypes = {
            site_header: 'str',
            event_header: 'str',
            hazard_header: 'float32'
        }
        cols = [event_header, site_header, hazard_header]
        hazard_input_df = pd.DataFrame(columns=cols)
        hazard_input_df[event_header] = hazard_scenario_list
        hazard_input_df[site_header] = '0'
        hazard_input_df[hazard_header] = hazard_intensity_list

        for col, dtype in dtypes.items():
            hazard_input_df[col] = hazard_input_df[col].astype(dtype)

        hazard_input_df.set_index([event_header, site_header], inplace=True, drop=True)
        hazard_input_df = hazard_input_df.unstack(level=-1, fill_value=0)
        hazard_input_df = hazard_input_df.droplevel(axis='columns', level=0)

        # return hazard_input_df, hazard_data_dict, hazard_scenario_list
        return hazard_input_df, hazard_scenario_list
