#!/usr/bin/env python

"""
Class definitions for sifra.py
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

def _readfile(setup_file):
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

class _IoDataGetter(object):
    """
    Class for reading in scenario setup information
    """
    def __init__(self, setup_file):
        self.setup = _readfile(setup_file)
        self.input_dir_name = self.setup["INPUT_DIR_NAME"]
        self.output_dir_name = self.setup["OUTPUT_DIR_NAME"]
        self.sys_config_file_name = self.setup["SYS_CONF_FILE_NAME"]
        self.input_path = None
        self.output_path = None
        self.root_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        self.raw_output_dir = None
        self.set_io_dirs()

    def set_io_dirs(self):
        self.input_path = os.path.join(self.root_dir,
                                       self.input_dir_name)
        self.output_path = os.path.join(self.root_dir,
                                        self.output_dir_name)
        self.raw_output_dir = os.path.join(self.root_dir,
                                           self.output_dir_name,
                                           'raw_output')

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(self.raw_output_dir):
            os.makedirs(self.raw_output_dir)


class _ScenarioDataGetter(object):
    """
    Class for reading in simulation scenario parameters
    """
    def __init__(self, setup_file):
        self.setup = _readfile(setup_file)
        self.fit_pe_data = self.setup["FIT_PE_DATA"]
        self.fit_restoration_data = self.setup["FIT_RESTORATION_DATA"]
        self.save_vars_npy = self.setup["SAVE_VARS_NPY"]
        self.intensity_measure_param = self.setup["INTENSITY_MEASURE_PARAM"]
        self.intensity_measure_unit = self.setup["INTENSITY_MEASURE_UNIT"]
        self.haz_param_min = self.setup["PGA_MIN"]
        self.haz_param_max = self.setup["PGA_MAX"]
        self.haz_param_step = self.setup["PGA_STEP"]
        self.num_samples = self.setup["NUM_SAMPLES"]
        self.time_unit = self.setup["TIME_UNIT"]
        self.run_parallel_proc = self.setup["MULTIPROCESS"]
        self.scenario_hazard_values = self.setup["SCENARIO_HAZARD_VALUES"]
        self.run_context = self.setup["RUN_CONTEXT"]
        self.restore_time_step = self.setup["RESTORE_TIME_STEP"]
        self.restore_pct_chkpoints = self.setup["RESTORE_PCT_CHKPOINTS"]
        self.restore_time_max = self.setup["RESTORE_TIME_MAX"]
        self.restoration_streams = self.setup["RESTORATION_STREAMS"]


class _RestorationDataGetter(object):
    """
    Class for reading in restoration scenario parameters
    """
    def __init__(self, setup_file):
        self.setup = _readfile(setup_file)
        self.restore_time_step = self.setup["RESTORE_TIME_STEP"]
        self.restore_pct_chkpoints = self.setup["RESTORE_PCT_CHKPOINTS"]
        self.restore_time_max = self.setup["RESTORE_TIME_MAX"]
        self.restoration_streams = self.setup["RESTORATION_STREAMS"]


class _FacilityDataGetter(object):
    """
    Module for reading in scenario setup information.
    It is a wrapper to protect the core classes from being affected by
    changes to variable names (e.g. if new units for hazard intensity
    are introduced), and changes to input file formats.
    """
    def __init__(self, setup_file):
        self.setup = _readfile(setup_file)
        self.system_classes = self.setup["SYSTEM_CLASSES"]
        self.system_class = self.setup["SYSTEM_CLASS"]
        self.system_subclass = self.setup["SYSTEM_SUBCLASS"]
        self.commodity_flow_types = self.setup["COMMODITY_FLOW_TYPES"]
        sys_config_file_name = self.setup["SYS_CONF_FILE_NAME"]
        self.input_dir_name = self.setup["INPUT_DIR_NAME"]
        self.sys_config_file = os.path.join(os.getcwd(),
                                            self.input_dir_name,
                                            sys_config_file_name)
        self.comp_df, self.fragility_data,\
        self.sysinp_setup, self.sysout_setup, self.node_conn_df = \
            self.assign_infrastructure_data()

        self.nominal_production = \
            self.sysout_setup['output_node_capacity'].sum()

        self.sys_dmg_states = ['DS0 None',
                               'DS1 Slight',
                               'DS2 Moderate',
                               'DS3 Extensive',
                               'DS4 Complete']

        # self.comp_df = pd.DataFrame()
        # self.node_conn_df = pd.DataFrame()
        """
        Set up data for:
        [1] system configuration data
        [2] component fragility algorithms
        Input data tables expected in the form of PANDAS DataFrames
        """

        self.comp_names = sorted(self.comp_df.index.tolist())
        self.num_elements = len(self.comp_names)

    def assign_infrastructure_data(self):
        config_file = self.sys_config_file
        NODE_CONN_DF = pd.read_excel(
            config_file, sheetname='component_connections',
            index_col=None, header=0,
            skiprows=3, skipinitialspace=True)

        COMP_DF = pd.read_excel(
            config_file, sheetname='component_list',
            index_col='component_id', header=0,
            skiprows=3, skipinitialspace=True)

        SYSOUT_SETUP = pd.read_excel(
            config_file, sheetname='output_setup',
            index_col='output_node', header=0,
            skiprows=3, skipinitialspace=True)
        SYSOUT_SETUP = SYSOUT_SETUP.sort_values(by='priority', ascending=True)

        SYSINP_SETUP = pd.read_excel(
            config_file, sheetname='supply_setup',
            index_col='input_node', header=0,
            skiprows=3, skipinitialspace=True)

        FRAGILITIES = pd.read_excel(
            config_file, sheetname='comp_type_dmg_algo',
            index_col=[0,1], header=0,
            skiprows=3, skipinitialspace=True)

        return COMP_DF, FRAGILITIES, SYSINP_SETUP, SYSOUT_SETUP, NODE_CONN_DF


# =============================================================================

class _Network(object):

    def __init__(self, facility):
        self.num_elements, self.G, self.nodes_all = \
            self.return_network(facility)

        self.sup_node_list, self.dep_node_list, \
        self.src_node_list, self.out_node_list = self.network_setup(facility)

    @staticmethod
    def return_network(facility):
        # ---------------------------------------------------------------------
        # Define the system as a network, with components as nodes
        # ---------------------------------------------------------------------
        comp_df = facility.comp_df
        node_conn_df = facility.node_conn_df
        nodes_all = sorted(comp_df.index)
        # nodes = comp_df.index.tolist()
        num_elements = len(nodes_all)

        #                    ------
        # Network setup with igraph (for analysis)
        #                    ------
        G = igraph.Graph(directed=True)

        G.add_vertices(len(nodes_all))
        G.vs["name"] = nodes_all
        G.vs["component_type"] = list(comp_df['component_type'].values)
        G.vs["cost_fraction"] = list(comp_df['cost_fraction'].values)
        G.vs["node_type"] = list(comp_df['node_type'].values)
        G.vs["node_cluster"] = list(comp_df['node_cluster'].values)
        G.vs["node_capacity"] = 1.0
        G.vs["functionality"] = 1.0

        for index, row in node_conn_df.iterrows():
            G.add_edge(row['origin'], row['destination'],
                       capacity=G.vs.find(row['origin'])["node_capacity"],
                       weight=row['weight'])
        return num_elements, G, nodes_all

    @staticmethod
    def network_setup(facility):
        #                    --------
        # Network setup with NetworkX (for drawing graph)
        #                    --------
        sys = nx.DiGraph()
        node_conn_df = facility.node_conn_df
        comp_df = facility.comp_df
        for index, row in node_conn_df.iterrows():
            sys.add_edge(row['origin'], row['destination'],
                       capacity=row['link_capacity'],
                       weight=row['weight'])

        if facility.system_class.lower() in ['potablewatertreatmentplant']:
            systemlayout.draw_sys_layout(
                sys, comp_df,
                out_dir=facility.output_path,
                graph_label="Water Treatment Plant Component Layout",
                orientation = "TB",
                connector_type="ortho",
                clustering=True
            )
        else:
            systemlayout.draw_sys_layout(
                sys, comp_df,
                out_dir=facility.output_path,
                graph_label="System Component Layout",
                orientation="LR",
                connector_type="spline",
                clustering=False
            )

        # ---------------------------------------------------------------------
        # List of tagged nodes with special roles:
        sup_node_list = [str(k) for k in
                         list(comp_df.ix[comp_df['node_type'] ==
                                         'supply'].index)]
        dep_node_list = [str(k) for k in
                         list(comp_df.ix[comp_df['node_type'] ==
                                         'dependency'].index)]
        src_node_list = [k for (k, v)in sys.in_degree().iteritems() if v == 0]
        out_node_list = \
            list(facility.sysout_setup.index.get_level_values('output_node'))

        return sup_node_list, dep_node_list, src_node_list, out_node_list


class FacilitySystem(_FacilityDataGetter, _IoDataGetter):
    """
    Defines an Critical Infrastructure Facility and its parameters
    """

    def __init__(self, setup_file):
        _FacilityDataGetter.__init__(self, setup_file)
        _IoDataGetter.__init__(self, setup_file)
        self.cp_types_in_system, self.cp_types_in_db = \
            self.check_types_with_db()
        self.uncosted_comptypes = \
            ['CONN_NODE', 'SYSTEM_INPUT', 'SYSTEM_OUTPUT']
        self.cp_types_costed, self.cpmap,\
            self.costed_comptypes, self.comps_costed = \
            self.list_of_components_for_cost_calculation()
        self.cpdict, self.output_dict,\
            self.input_dict, self.nodes_by_commoditytype = \
            self.convert_df_to_dict()
        self.sys = self.build_system_model()
        self.fragdict = self.fragility_dict()
        self.compdict = self.comp_df.to_dict()
        self.network = _Network(self)

    def build_system_model(self):
        """
        Build the network model of the system
        Uses igraph for generating the system config as a graph/network.
        igraph was used for speed as it is written in C.
        """
        sys = igraph.Graph(directed=True)

        sys.add_vertices(len(self.comp_names))
        sys.vs["name"] = self.comp_names

        sys.vs["component_type"] = \
            list(self.comp_df['component_type'].values)
        sys.vs["cost_fraction"] = \
            list(self.comp_df['cost_fraction'].values)

        sys.vs["node_type"] = \
            list(self.comp_df['node_type'].values)
        sys.vs["node_cluster"] = \
            list(self.comp_df['node_cluster'].values)

        sys.vs["node_capacity"] = 1.0
        sys.vs["functionality"] = 1.0

        for _, row in self.node_conn_df.iterrows():
            sys.add_edge(
                row['origin'], row['destination'],
                capacity=sys.vs.find(row['origin'])['node_capacity'],
                weight=row['weight'])
        return sys

    def convert_df_to_dict(self):
        # ---------------------------------------------------------------------
        # Convert Dataframes to Dicts for lookup efficiency
        # ---------------------------------------------------------------------
        comp_df = self.comp_df
        sysout_setup = self.sysout_setup
        sysinp_setup = self.sysinp_setup

        cpdict = {}
        for i in list(comp_df.index):
            cpdict[i] = comp_df.ix[i].to_dict()

        output_dict = {}
        outputnodes = \
            list(np.unique(sysout_setup.index.get_level_values('output_node')))

        for k1 in outputnodes:
            output_dict[k1] = {}
            output_dict[k1] = sysout_setup.ix[k1].to_dict()

        input_dict = {}
        inputnodes = \
            list(np.unique(sysinp_setup.index.get_level_values('input_node')))

        for k1 in inputnodes:
            input_dict[k1] = {}
            input_dict[k1] = sysinp_setup.ix[k1].to_dict()

        nodes_by_commoditytype = {}
        for i in np.unique(sysinp_setup['commodity_type']):
            nodes_by_commoditytype[i] \
                = [x for x in sysinp_setup.index
                   if sysinp_setup.ix[x]['commodity_type'] == i]

        return cpdict, output_dict, input_dict, nodes_by_commoditytype

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

    def fragility_dict(self):
        fragilities = self.fragility_data
        for comp in fragilities.index.levels[0]:
            fragilities.loc[(comp, 'DS0 None'), 'damage_function'] = 'Lognormal'
            fragilities.loc[(comp, 'DS0 None'), 'damage_median'] = np.inf
            fragilities.loc[(comp, 'DS0 None'), 'damage_logstd'] = 1.0
            fragilities.loc[(comp, 'DS0 None'), 'damage_lambda'] = 0.01
            fragilities.loc[(comp, 'DS0 None'), 'damage_ratio'] = 0.0
            fragilities.loc[(comp, 'DS0 None'), 'recovery_mean'] = -np.inf
            fragilities.loc[(comp, 'DS0 None'), 'recovery_std'] = 1.0
            fragilities.loc[(comp, 'DS0 None'), 'functionality'] = 1.0
            fragilities.loc[(comp, 'DS0 None'), 'mode'] = 1
            fragilities.loc[(comp, 'DS0 None'), 'minimum'] = -np.inf
            fragilities.loc[(comp, 'DS0 None'), 'sigma_1'] = 'NA'
            fragilities.loc[(comp, 'DS0 None'), 'sigma_2'] = 'NA'

        fgdt = fragilities.to_dict()
        fragdict = {}
        for key, val in fgdt.iteritems():
            elemdict = {}
            for t, v in val.iteritems():
                elem = t[0]
                ds = t[1]
                if elem not in elemdict.keys():
                    elemdict[elem] = {}
                    elemdict[elem][ds] = v
                elif ds not in elemdict[elem].keys():
                    elemdict[elem][ds] = v
            fragdict[key] = elemdict

        return fragdict

    def check_types_with_db(self):
        comp_df = self.comp_df
        fragilities = self.fragility_data
        # check to ensure component types match with DB
        cp_types_in_system = \
            list(np.unique(comp_df['component_type'].tolist()))
        cp_types_in_db = list(fragilities.index.levels[0])
        assert set(cp_types_in_system).issubset(cp_types_in_db) == True
        return cp_types_in_system, cp_types_in_db

    def list_of_components_for_cost_calculation(self):
        # get list of only those components that are included in
        # cost calculations
        cp_types_costed = [x for x in self.cp_types_in_system
                           if x not in self.uncosted_comptypes]
        costed_comptypes = sorted(list(set(self.cp_types_in_system) -
                                       set(self.uncosted_comptypes)))
        cpmap = {c: sorted(self.comp_df[self.comp_df['component_type'] == c].
                           index.tolist()) for c in self.cp_types_in_system}
        comps_costed = [v for x in cp_types_costed for v in cpmap[x]]
        return cp_types_costed, cpmap, costed_comptypes, comps_costed

    # def draw_layout(self, facility):
    #     """
    #     Generate a diagram with the graph layout representing
    #     the system configuration
    #     The directed graph is generated with NetworkX
    #     """
    #     out_dir = self.output_path
    #     grx = nx.DiGraph()
    #     for _, row in self.node_conn_df.iterrows():
    #         grx.add_edge(row['origin'], row['destination'],
    #                      capacity=row['link_capacity'],
    #                      weight=row['weight'])
    #     systemlayout.draw_sys_layout(
    #         grx, self.comp_df,
    #         out_dir=out_dir,
    #         out_file="system_layout",
    #         graph_label="System Component Layout")


class Scenario(_ScenarioDataGetter, _IoDataGetter):
    """
    Defines the scenario for hazard impact modelling
    """

    def __init__(self, setup_file):
        _ScenarioDataGetter.__init__(self, setup_file)
        _IoDataGetter.__init__(self, setup_file)

        """Set up parameters for simulating hazard impact"""
        self.num_hazard_pts = \
            int(round((self.haz_param_max - self.haz_param_min) /
                      float(self.haz_param_step) + 1))

        self.hazard_intensity_vals = \
            np.linspace(self.haz_param_min, self.haz_param_max,
                        num=self.num_hazard_pts)
        self.hazard_intensity_str = \
            [('%0.3f' % np.float(x)) for x in self.hazard_intensity_vals]

        # Set up parameters for simulating recovery from hazard impact
        self.restoration_time_range, self.time_step = np.linspace(
            0, self.restore_time_max, num=self.restore_time_max + 1,
            endpoint=True, retstep=True)

        self.num_time_steps = len(self.restoration_time_range)

        self.restoration_chkpoints, self.restoration_pct_steps = \
            np.linspace(0.0, 1.0, num=self.restore_pct_chkpoints, retstep=True)


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

        ct_fragility_df = fragility_data.loc[(self.type,), dmg_headers].\
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


# *****************************************************************************
# Definitions of specific infrastructure systems:
# These are variations on a Facility or a Network
# *****************************************************************************

class PowerStation(FacilitySystem):
    """
    Defines specific attributes of a power station, customising
    the Critical Infrastructure Facility class
    """

    def __init__(self, setup_file):
        super(PowerStation, self).__init__(setup_file)

        self.asset_type = 'Power Station'
        self.name = ''
        generation_tech = self.system_subclass

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

class PotableWaterTreatmentPlant(FacilitySystem):
    """
    Defines specific attributes of a potable water treatment plant,
    customising the Critical Infrastructure Facility class
    """

    def __init__(self, setup_file):
        super(PotableWaterTreatmentPlant, self).__init__(setup_file)

        self.asset_type = "Potable Water Treatment Plant"
        self.name = ''

        self.sys_dmg_states = ['DS0 None',
                               'DS1 Slight',
                               'DS2 Moderate',
                               'DS3 Extensive',
                               'DS4 Complete']

        self.dmg_scale_criteria = 'Economic Loss'
        self.dmg_scale_bounds = [0.10, 0.30, 0.50, 0.75, 1.00]

# =============================================================================
