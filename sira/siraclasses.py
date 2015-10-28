#!/usr/bin/env python

"""
Class definitions for sira.py
"""

from __future__ import print_function
import os
import numpy as np
import pandas as pd
import igraph
import networkx as nx
import systemlayout
import matplotlib.pyplot as plt
import seaborn as sns
import re
import copy
from scipy import stats

# =============================================================================

def readfile(setup_file):
    """
    Module for reading in scenario data file
    """
    discard = {}
    setup = {}
    # try:
    #     with open(setup_file) as fh:
    #         exec(fh.read(), discard, setup)
    # except IOError as err:
    #     print("{0}".format(err))
    #     raise SystemExit()
    if not os.path.isfile(setup_file):
        print("[ERROR] could not read file: {}".format(setup_file))
        raise SystemExit()
    exec (open(setup_file).read(), discard, setup)
    return setup


# =============================================================================

class IoDataGetter(object):
    """
    Class for reading in scenario setup information
    """
    def __init__(self, setup_file):
        self.setup = readfile(setup_file)
        self.input_dir_name = self.setup["INPUT_DIR_NAME"]
        self.output_dir_name = self.setup["OUTPUT_DIR_NAME"]
        self.sys_config_file_name = self.setup["SYS_CONF_FILE_NAME"]
        self.input_path = None
        self.output_path = None
        self.raw_output_dir = None
        self.set_io_dirs()

    def set_io_dirs(self):
        self.input_path = os.path.join(os.getcwd(), self.input_dir_name)

        if not os.path.exists(self.output_dir_name):
            os.makedirs(self.output_dir_name)

        self.output_path = \
            os.path.join(os.getcwd(), self.output_dir_name)

        self.raw_output_dir = \
            os.path.join(os.getcwd(), self.output_dir_name,
                         'raw_output')

        if not os.path.exists(self.raw_output_dir):
            os.makedirs(self.raw_output_dir)


class ScenarioDataGetter(object):
    """
    Class for reading in simulation scenario parameters
    """
    def __init__(self, setup_file):
        self.setup = readfile(setup_file)
        self.fit_pe_data = self.setup["FIT_PE_DATA"]
        self.fit_restoration_data = self.setup["FIT_RESTORATION_DATA"]
        self.save_vars_npy = self.setup["SAVE_VARS_NPY"]
        self.hazard_transfer_param = self.setup["HAZARD_TRANSFER_PARAM"]
        self.hazard_transfer_unit = self.setup["HAZARD_TRANSFER_UNIT"]
        self.haz_param_min = self.setup["PGA_MIN"]
        self.haz_param_max = self.setup["PGA_MAX"]
        self.haz_param_step = self.setup["PGA_STEP"]
        self.num_samples = self.setup["NUM_SAMPLES"]
        self.time_unit = self.setup["TIME_UNIT"]
        self.restore_time_step = self.setup["RESTORE_TIME_STEP"]
        self.restore_pct_chkpoints = self.setup["RESTORE_PCT_CHKPOINTS"]
        self.restore_time_upper = self.setup["RESTORE_TIME_UPPER"]
        self.restore_time_max = self.setup["RESTORE_TIME_MAX"]


class FacilityDataGetter(object):
    """
    Module for reading in scenario setup information
    It is a wrapper to protect the core classes from being affected by
    changes to variable names (e.g. if new units for hazard intensity
    are introduced), and changes to input file formats.
    """
    def __init__(self, setup_file):
        self.setup = readfile(setup_file)
        self.system_classes = self.setup["SYSTEM_CLASSES"]
        self.system_class = self.setup["SYSTEM_CLASS"]
        self.commodity_flow_types = self.setup["COMMODITY_FLOW_TYPES"]
        self.sys_config_file_name = self.setup["SYS_CONF_FILE_NAME"]
        self.input_dir_name = self.setup["INPUT_DIR_NAME"]

        self.sys_config_file = None
        self.fragility_data = pd.DataFrame()
        self.sysinp_setup = pd.DataFrame()
        self.node_conn_df = pd.DataFrame()
        self.sysout_setup = pd.DataFrame()
        self.assign_infrastructure_data()

    def assign_infrastructure_data(self):
        """
        Assign parameters that define the fragility and restoration
        of components that constitute the infrastructure facility
        that is the subject of the simulation
        """
        self.sys_config_file = os.path.join(
            os.getcwd(), self.input_dir_name,
            self.sys_config_file_name)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # List of components in the system, with specified attributes
        self.comp_df = pd.read_excel(
            self.sys_config_file, 'comp_list',
            index_col='component_id',
            skiprows=3, skipinitialspace=True)

        self.comp_df = self.comp_df.rename(
            columns={'component_id': 'component_id',
                     'component_type': 'component_type',
                     'component_class': 'component_class',
                     'cost_fraction': 'cost_fraction',
                     'node_type': 'node_type',
                     'node_cluster': 'node_cluster',
                     'op_capacity': 'op_capacity'
                     })

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Connections between the components (nodes), with edge attributes
        self.node_conn_df = pd.read_excel(
            self.sys_config_file, 'component_connections',
            index_col=None, skiprows=3, skipinitialspace=True)

        self.node_conn_df = self.node_conn_df.rename(
            columns={'Orig': 'Origin',
                     'Dest': 'Destination',
                     'Capacity': 'Capacity',
                     'Weight': 'Weight',
                     'Distance': 'Distance'
                     })

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Configuration of output nodes (sinks)
        self.sysout_setup = pd.read_excel(
            self.sys_config_file, 'output_setup',
            index_col='OutputNode',
            skiprows=3, skipinitialspace=True)

        self.sysout_setup = self.sysout_setup.rename(
            columns={'OutputNode': 'Output Node',
                     'ProductionNode': 'Production Node',
                     'Capacity': 'Capacity',
                     'CapFraction': 'Capacity Fraction',
                     'Priority': 'Priority'
                     })
        self.sysout_setup = self.sysout_setup.sort(
            'Priority', ascending=True)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Configuration of input nodes
        # i.e. supply of 'commodities' into the system
        self.sysinp_setup = pd.read_excel(
            self.sys_config_file, 'supply_setup',
            index_col='InputNode',
            skiprows=3, skipinitialspace=True)

        self.sysinp_setup = self.sysinp_setup.rename(
            columns={'InputNode': 'Input Node',
                     'Capacity': 'Capacity',
                     'CapFraction': 'Capacity Fraction',
                     'CommodityType': 'Commodity Type'
                     })

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Component fragility and recovery algorithms, by component type
        self.fragility_data = pd.read_excel(
            self.sys_config_file, 'comp_type_dmg_algo',
            index_col=['component_type', 'damage_state'],
            skiprows=3, skipinitialspace=True)

        self.comp_df = self.comp_df.rename(
            index={'component_type': 'component_type',
                   'damage_state': 'damage_state'
                   },
            columns={'damage_function': 'damage_function',
                     'function_mode': 'function_mode',
                     'damage_median': 'damage_median',
                     'damage_logstd': 'damage_logstd',
                     'damage_ratio': 'damage_ratio',
                     'functionality': 'functionality',
                     'minimum': 'minimum',
                     'sigma_1': 'sigma_1',
                     'sigma_2': 'sigma_2',
                     'recovery_mean': 'recovery_mean',
                     'recovery_std': 'recovery_std',
                     'fragility_source': 'fragility_source'
                     })


# =============================================================================

class Scenario(object):
    """
    Defines the scenario for hazard impact modelling
    """

    def __init__(self, setup_file):
        self.data = ScenarioDataGetter(setup_file)
        self.io = IoDataGetter(setup_file)

        self.input_dir_name = self.io.input_dir_name
        self.output_dir_name = self.io.output_dir_name
        # self.fit_pe_data = self.data.fit_pe_data
        # self.fit_restoration_data = self.data.fit_restoration_data
        # self.save_vars_npy = self.data.save_vars_npy

        self.input_path = self.io.input_path
        self.output_path = self.io.output_path
        self.raw_output_dir = self.io.raw_output_dir

        # self.set_io_dirs()
        self.set_hazard_params()
        self.set_restoration_params()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def set_hazard_params(self):
        """
        Set up parameters for simulating hazard impact
        """
        self.hazard_transfer_param = self.data.hazard_transfer_param
        self.hazard_transfer_unit = self.data.hazard_transfer_unit

        self.pga_min = self.data.haz_param_min
        self.pga_max = self.data.haz_param_max
        self.pga_step = self.data.haz_param_step
        self.num_samples = self.data.num_samples

        self.num_hazard_pts = int(round((self.pga_max - self.pga_min) /
                                        float(self.pga_step) + 1))

        # FORMERLY: PGA_levels
        self.hazard_intensity_vals = np.linspace(self.pga_min, self.pga_max,
                                                 num=self.num_hazard_pts)

    def set_restoration_params(self):
        """
        Set up parameters for simulating recovery from hazard impact
        """
        self.time_unit = self.data.time_unit
        self.restore_time_step = self.data.restore_time_step
        self.restore_pct_chkpoints = self.data.restore_pct_chkpoints
        self.restore_time_upper = self.data.restore_time_upper
        self.restore_time_max = self.data.restore_time_max

        self.restoration_time_range, self.time_step = \
            np.linspace(0, self.restore_time_upper,
                        num=self.restore_time_upper + 1,
                        endpoint=True, retstep=True)

        self.num_time_steps = len(self.restoration_time_range)

        self.restoration_chkpoints, self.restoration_pct_steps = \
            np.linspace(0.0, 1.0,
                        num=self.restore_pct_chkpoints,
                        retstep=True)


# =============================================================================

class Facility(object):
    """
    Defines an Critical Infrastructure Facility and its parameters
    """

    def __init__(self, setup_file):
        # self.setup = readfile(setup_file)
        self.io = IoDataGetter(setup_file)
        self.data = FacilityDataGetter(setup_file)

        self.system_class = self.data.system_class
        self.commodity_flow_types = self.data.commodity_flow_types

        self.sys_config_file = os.path.join(
            os.getcwd(),
            self.io.input_dir_name,
            self.data.sys_config_file_name)

        self.sys_dmg_states = ['DS0 None',
                               'DS1 Slight',
                               'DS2 Moderate',
                               'DS3 Extensive',
                               'DS4 Complete']

        self.comp_df = pd.DataFrame()
        self.node_conn_df = pd.DataFrame()
        self.setup_system_data()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def setup_system_data(self):
        """
        Set up data for:
        [1] system configuration data
        [2] component fragility algorithms
        Input data tables expected in the form of PANDAS DataFrames
        """

        # List of components in the system, with specified attributes
        self.comp_df = self.data.comp_df

        # Connections between the components (nodes), with edge attributes
        self.node_conn_df = self.data.node_conn_df

        # Configuration of output nodes (sinks)
        sysout_setup = self.data.sysout_setup
        self.sysout_setup = sysout_setup.sort('Priority', ascending=True)

        # Configuration of input nodes supplying 'commodities' into the system
        self.sysinp_setup = self.data.sysinp_setup

        # Component fragility and recovery algorithms, by component type
        self.fragility_data = self.data.fragility_data

        self.comp_names = sorted(self.comp_df.index.tolist())
        self.num_elements = len(self.comp_names)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def build_system_model(self):
        """
        Build the network model of the system
        Uses igraph for generating the system config as a graph/network.
        igraph was used for speed as it is written in C.
        """
        self.sys = igraph.Graph(directed=True)
        self.sys.add_vertices(len(self.comp_names))
        self.sys.vs["name"] = self.comp_names

        self.sys.vs["component_type"] = \
            list(self.comp_df['component_type'].values)
        self.sys.vs["cost_fraction"] = \
            list(self.comp_df['cost_fraction'].values)

        self.sys.vs["node_type"] = \
            list(self.comp_df['node_type'].values)
        self.sys.vs["node_cluster"] = \
            list(self.comp_df['node_cluster'].values)

        self.sys.vs["capacity"] = 1.0
        self.sys.vs["functionality"] = 1.0

        for _, row in self.node_conn_df.iterrows():
            self.sys.add_edge(
                row['Origin'], row['Destination'],
                capacity=self.sys.vs.find(row['Origin'])['capacity'],
                weight=row['Weight'],
                distance=row['Distance'])

            # return self.sys

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def add_component(self, component_id):
        """
        Add member component / element to the system
        """
        comp_type = self.comp_df.loc[component_id, 'component_type']
        comp_obj = ComponentType(comp_type, self.fragility_data)
        comp_obj.name = component_id
        # dmg_states =\
        #     sorted([str(d_s) for d_s in self.fragility_data.index.levels[1]])
        return comp_obj

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def draw_layout(self):
        """
        Generate a diagram with the graph layout representing
        the system configuration
        The directed graph is generated with NetworkX
        """
        out_dir = self.io.output_path
        grx = nx.DiGraph()
        for _, row in self.node_conn_df.iterrows():
            grx.add_edge(row['Origin'], row['Destination'],
                         capacity=row['Capacity'],
                         weight=row['Weight'],
                         distance=row['Distance'])
        systemlayout.draw_sys_layout(
            grx, self.comp_df,
            out_dir=out_dir,
            out_file="system_layout",
            graph_label="System Component Layout")


# =============================================================================


class ComponentType(object):
    """
    Represents the <Component Types> that comprise the System
    """

    def __init__(self, comp_type, fragility_data):

        self.type = comp_type
        # fragility = {}
        # restoration = {}

        dmg_headers = ['damage_function', 'function_mode', 'minimum',
                       'damage_median', 'damage_logstd',
                       'damage_ratio', 'functionality',
                       'sigma_1', 'sigma_2',
                       'recovery_mean', 'recovery_std']

        ct_fragility_df = fragility_data.loc[(self.type,), dmg_headers]. \
            query("damage_state!='DS0 None'")

        # for elem in ct_fragility_df.index.tolist():
        #     self.fragility[elem] = ct_fragility_df.ix[elem].to_dict()
        self.fragility_dict = ct_fragility_df.to_dict()

        self.damage_function = {}
        self.damage_function_mode = {}
        self.damage_function_min = {}
        self.damage_median = {}
        self.damage_logstd = {}
        self.damage_ratio = {}
        self.damage_function_sigma1 = {}
        self.damage_function_sigma2 = {}

        self.functionality = {}
        self.recovery_mean = {}
        self.recovery_std = {}

        self.add_fragility_algorithm()
        self.add_restoration_params()
        self.add_none_damage_state()

        # These attributes are to set by instances,
        # within the context of the system:
        self.cost_fraction = {}
        self.node_type = {}
        self.node_cluster = {}

        # ??? CHECK: same as 'functionality'?
        self.op_capacity = {}

        self.node_obj = None  # <--- this is the hook into the graph

    def add_none_damage_state(self):
        """
        Add 'DS0 None' damage state
        """
        self.damage_function['DS0 None'] = 'Lognormal'
        self.damage_function_min['DS0 None'] = -np.inf
        self.damage_function_mode['DS0 None'] = 1

        self.damage_median['DS0 None'] = np.inf
        self.damage_logstd['DS0 None'] = 1.0
        self.damage_ratio['DS0 None'] = 0.0

        self.damage_function_sigma1['DS0 None'] = 'NA'
        self.damage_function_sigma2['DS0 None'] = 'NA'

        self.functionality = 1.0
        self.recovery_mean = -np.inf
        self.recovery_std = 1.0

    def add_fragility_algorithm(self):
        """
        Add fragility algorithm for the given component type
        """
        self.damage_function = self.fragility_dict['damage_function']
        self.damage_function_min = self.fragility_dict['minimum']
        self.damage_function_mode = self.fragility_dict['function_mode']
        self.damage_median = self.fragility_dict['damage_median']
        self.damage_logstd = self.fragility_dict['damage_logstd']
        self.damage_ratio = self.fragility_dict['damage_ratio']
        self.damage_function_sigma1 = self.fragility_dict['sigma_1']
        self.damage_function_sigma2 = self.fragility_dict['sigma_2']

    def add_restoration_params(self):
        """
        Add restoration parameters for the given component type
        """
        self.functionality = self.fragility_dict['functionality']
        self.recovery_mean = self.fragility_dict['recovery_mean']
        self.recovery_std = self.fragility_dict['recovery_std']

    def plot_comp_frag(self, fragility_data, hazard_intensity_vals,
                       fig_output_path='./component_fragilities'):
        """
        Generates fragility plots of given component types, and
        saves them in <OUTPUT_PATH>
        """
        dsclrs = ["#2ecc71", "#3498db", "#feb24c", "#de2d26"]
        grid_colr = '#B6B6B6'

        sns.set_style('whitegrid')
        sns.set_palette(dsclrs)
        sns.set_context('paper')

        if not os.path.exists(fig_output_path):
            os.makedirs(fig_output_path)

        fragility_df = copy.deepcopy(fragility_data)
        fragility_df = fragility_df.query("damage_state!='DS0 None'")

        ds_list = list(fragility_data.index.levels[1].values)
        if 'DS0 None' in ds_list:
            ds_list.remove('DS0 None')

        frag_fig = plt.figure(figsize=(6.0, 3.2))
        axis = frag_fig.add_subplot(111)
        for dmg_state in ds_list:
            scale = fragility_data.loc[(self.type, dmg_state), 'damage_median']
            beta = fragility_data.loc[(self.type, dmg_state), 'damage_logstd']
            plt.plot(
                hazard_intensity_vals,
                stats.lognorm.cdf(hazard_intensity_vals, beta, scale=scale),
                clip_on=False)

        plt.grid(True, which='major', axis='both',
                 linestyle='-', linewidth=0.5, color=grid_colr)
        plt.title(self.type, loc='center', y=1.02, size=8)

        plt.xlabel('PGA (g)', size=7, labelpad=8)
        plt.ylabel('Probability of Exceedence', size=7, labelpad=8)
        plt.yticks(np.linspace(0.0, 1.0, 11))

        axis.tick_params(
            axis='both',
            which='both',
            left='off',
            right='off',
            labelleft='on',
            labelright='off',
            direction='out',
            labelsize=6,
            pad=5)

        box = axis.get_position()
        axis.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        axis.legend(ds_list, loc='upper left', ncol=1,
                    bbox_to_anchor=(1.02, 1.0),
                    frameon=0, prop={'size': 6})

        cpname_cleaned = re.sub(r"/|\\|\*", "_", self.type)
        plt.savefig(os.path.join(fig_output_path, cpname_cleaned + '.png'),
                    format='png', bbox_inches='tight', dpi=300)
        plt.close()


# =============================================================================

class PowerStation(Facility):
    """
    Defines specific attributes of a power station, customising
    the Critical Infrastructure Facility class
    """

    def __init__(self, setup_file, generation_tech=''):
        super(PowerStation, self).__init__(setup_file)

        self.asset_type = 'Power Station'
        self.name = ''

        self.sys_dmg_states = ['DS0 None',
                               'DS1 Slight',
                               'DS2 Moderate',
                               'DS3 Extensive',
                               'DS4 Complete']

        if generation_tech.lower() == 'coal fired':
            self.dmg_scale_criteria = 'Economic Loss'
            self.dmg_scale_bounds = [0.01, 0.15, 0.4, 0.8, 1.0]

        elif generation_tech.lower() == 'combined cycle':
            self.dmg_scale_criteria = 'Economic Loss'
            self.dmg_scale_bounds = [0.01, 0.15, 0.4, 0.8, 1.0]

# =============================================================================
