#!/usr/bin/env python
# # -*- coding: utf-8 -*-
"""
title           : __main__.py
description     : entry point for core sifra component
usage           : python sifra [OPTIONS]
                  -s                Display this usage message
                  -l [LEVEL]        Choose logging level DEBUG, INFO,
                                    WARNING, ERROR, CRITICAL

python_version  : 2.7
"""

from __future__ import print_function

import numpy as np
np.seterr(divide='print', invalid='raise')

import pandas as pd

import copy
from colorama import Fore, Back, init, Style
init()

import seaborn as sns
sns.set(style='whitegrid', palette='coolwarm')

import os
import argparse
from sifra.logger import rootLogger
from sifra.configuration import Configuration
from sifra.scenario import Scenario
from sifra.modelling.hazard import HazardsContainer
from sifra.model_ingest import ingest_model
from sifra.simulation import calculate_response
from sifra.modelling.system_topology import SystemTopology
from sifra.infrastructure_response import (
    write_system_response,
    plot_mean_econ_loss,
    pe_by_component_class
    )
from sifra import fit_model

from sifra.loss_analysis import (draw_component_loss_barchart_s1,
                                 draw_component_loss_barchart_s2,
                                 draw_component_loss_barchart_s3,
                                 draw_component_failure_barchart,
                                 calc_comptype_damage_scenario_given_hazard,
                                 prep_repair_list,
                                 calc_restoration_setup,
                                 vis_restoration_process,
                                 component_criticality)

import numpy as np

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-o", "--output", type=str)

    parser.add_argument("-v", "--verbose",  type=str,
                        help="Choose option for logging level from: \n"
                             "DEBUG, INFO, WARNING, ERROR, CRITICAL.")
    args = parser.parse_args()

    rootLogger.set_log_level(args.verbose)

    if args.config is not None and args.model is not None and args.output is not None:

        rootLogger.set_log_file_path(os.path.join(args.output, "log.txt"))
        rootLogger.info('Simulation initiated...')

        # ---------------------------------------------------------------------
        # Configure simulation model.
        # Read data and control parameters and construct objects.
        # ---------------------------------------------------------------------

        config = Configuration(args.config, args.model, args.output)
        scenario = Scenario(config)
        hazards = HazardsContainer(config)
        infrastructure = ingest_model(config)

        # ---------------------------------------------------------------------
        # Run simulation.
        # Get the results of running a simulation
        # ---------------------------------------------------------------------
        response_list = [
            {},  # hazard level vs component damage state index
            {},  # hazard level vs infrastructure output
            {},  # hazard level vs component response
            {},  # hazard level vs component type response
            [],  # array of infrastructure output per sample
            [],  # array infrastructure econ loss per sample
            {},  # hazard level vs component class dmg level pct
            {}]  # hazard level vs component class expected damage index

        # response_list = calculate_response(hazards, scenario, infrastructure)
        #
        # # ---------------------------------------------------------------------
        # # Post simulation processing.
        # # After the simulation has run the results are aggregated, saved
        # # and the system fragility is calculated.
        #
        # write_system_response(response_list, infrastructure, scenario, hazards)
        # economic_loss_array = response_list[5]
        # plot_mean_econ_loss(scenario, economic_loss_array, hazards)
        #
        # if config.HAZARD_INPUT_METHOD == "hazard_array":
        #     pe_by_component_class(response_list, infrastructure,scenario, hazards)
        #
        # # ---------------------------------------------------------------------
        # # Visualizations
        # # Construct visualization for system topology
        # # ---------------------------------------------------------------------
        # sys_topology_view = SystemTopology(infrastructure, scenario)
        # sys_topology_view.draw_sys_topology(viewcontext="as-built")
        #
        # rootLogger.info('Simulation completed...')


        write_system_response(response_list, infrastructure, scenario, hazards)
        economic_loss_array = response_list[5]
        plot_mean_econ_loss(scenario, economic_loss_array, hazards)
        rootLogger.info("End serial run")
        if config.HAZARD_INPUT_METHOD == "hazard_array":
            pe_by_component_class(response_list, infrastructure,scenario, hazards)

        # ---------------------------------------------------------------------
        # FIT MODEL ANALYSIS
        # ---------------------------------------------------------------------
        rootLogger.info('Start: FIT MODEL ANALYSIS')

        hazard_scenarios = hazards.hazard_scenario_list
        sys_limit_states = infrastructure.get_system_damage_states()
        FIT_PE_DATA = scenario.fit_pe_data

        existing_models = ["potablewatertreatmentplant", "pwtp", "wastewatertreatmentplant", "wwtp", "watertreatmentplant", "wtp"]

        if infrastructure.system_class.lower() == 'powerstation':
            pe_sys = np.load(os.path.join(config.RAW_OUTPUT_DIR, 'pe_sys_econloss.npy'))
        elif infrastructure.system_class.lower() == 'substation':
            pe_sys = np.load(os.path.join(config.RAW_OUTPUT_DIR, 'pe_sys_cpfailrate.npy'))
        elif infrastructure.system_class.lower() in existing_models:
            pe_sys = np.load(os.path.join(config.RAW_OUTPUT_DIR, 'pe_sys_econloss.npy'))

        if FIT_PE_DATA:
            # Calculate & Plot Fitted Models
            fit_model.fit_prob_exceed_model(hazard_scenarios, pe_sys, sys_limit_states, config.OUTPUT_PATH, config)

        rootLogger.info('End: FIT MODEL ANALYSIS')


        # ---------------------------------------------------------------------
        # SCENARIO LOSS ANALYSIS
        # ---------------------------------------------------------------------
        rootLogger.info('Start: SCENARIO LOSS ANALYSIS')

        RESTORATION_STREAMS = scenario.restoration_streams
        FOCAL_HAZARD_SCENARIOS = hazards.focal_hazard_scenarios

        # Restoration time starts x time units after hazard impact:
        # This represents lead up time for damage and safety assessments
        RESTORATION_OFFSET = 1

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Read in SIMULATED HAZARD RESPONSE for <COMPONENT TYPES>

        comptype_resp_df = pd.read_csv(os.path.join(config.OUTPUT_PATH, 'comptype_response.csv'), index_col=['component_type', 'response'], skipinitialspace=True)
        comptype_resp_df.columns = hazards.hazard_scenario_name

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Nodes not considered in the loss calculations
        # NEED TO MOVE THESE TO A MORE LOGICAL PLACE

        uncosted_comptypes = ['CONN_NODE', 'SYSTEM_INPUT', 'SYSTEM_OUTPUT', 'JUNCTION_NODE', 'JUNCTION', 'Generation Source', 'Grounding']

        cp_types_in_system = infrastructure.get_component_types()
        cp_types_costed = [x for x in cp_types_in_system if x not in uncosted_comptypes]

        comptype_resp_df = comptype_resp_df.drop(uncosted_comptypes, level='component_type', axis=0)

        # Get list of only those components that are included in cost calculations:
        cpmap = {c: sorted(list(infrastructure.get_components_for_type(c))) for c in cp_types_in_system}
        comps_costed = [v for x in cp_types_costed for v in cpmap[x]]

        nodes_all = infrastructure.components.keys()
        nodes_all.sort()
        comps_uncosted = list(set(nodes_all).difference(comps_costed))

        ctype_failure_mean = comptype_resp_df.xs('num_failures', level='response')
        ctype_failure_mean.columns.names = ['Scenario Name']

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Value of component types relative to system value

        comptype_value_dict = {}
        for ct in sorted(cp_types_costed):
            comp_val = [infrastructure.components[comp_id].cost_fraction for comp_id in cpmap[ct]]
            comptype_value_dict[ct] = sum(comp_val)

        comptype_value_list = [comptype_value_dict[ct] for ct in sorted(comptype_value_dict.keys())]
        component_response = pd.read_csv(os.path.join(config.OUTPUT_PATH, 'component_response.csv'), index_col=['component_id', 'response'], skiprows=0, skipinitialspace=True)
        component_meanloss = component_response.query('response == "loss_mean"').reset_index('response').drop('response', axis=1)
        comptype_resp_df = comptype_resp_df.drop(uncosted_comptypes, level='component_type', axis=0)

        weight_criteria = 'MIN_COST'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        col_tp = []
        for h in FOCAL_HAZARD_SCENARIOS:
            col_tp.extend(zip([h] * len(RESTORATION_STREAMS), RESTORATION_STREAMS))

        mcols = pd.MultiIndex.from_tuples(col_tp, names=['Hazard', 'Restoration Streams'])
        time_to_full_restoration_for_lines_df = pd.DataFrame(index=infrastructure.output_nodes.keys(), columns=mcols)
        time_to_full_restoration_for_lines_df.index.name = 'Output Lines'

        # --------------------------------------------------------------------------
        # *** BEGIN : FOCAL_HAZARD_SCENARIOS FOR LOOP ***
        for sc_haz_str in FOCAL_HAZARD_SCENARIOS:

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Differentiated setup based on hazard input type - scenario vs array
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            sc_haz_str = "{:.3f}".format(float(sc_haz_str))
            if config.HAZARD_INPUT_METHOD == "hazard_array":
                scenario_header = hazards.hazard_scenario_name[hazards.hazard_scenario_list.index(sc_haz_str)]
            elif config.HAZARD_INPUT_METHOD == "scenario_file":
                scenario_header = sc_haz_str
            scenario_tag = str(sc_haz_str) + " " + hazards.intensity_measure_unit + " " + hazards.intensity_measure_param

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Extract scenario-specific values from the 'hazard response' dataframe
            # Scenario response: by component type
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            ctype_resp_scenario = comptype_resp_df[scenario_header].unstack(level=-1)
            ctype_resp_scenario = ctype_resp_scenario.sort_index()

            ctype_resp_scenario['loss_per_type'] = ctype_resp_scenario['loss_mean'] / comptype_value_list

            ctype_resp_scenario['loss_per_type_std'] = ctype_resp_scenario['loss_std'] * [len(list(infrastructure.get_components_for_type(ct))) for ct in ctype_resp_scenario.index.values.tolist()]

            ctype_resp_sorted = ctype_resp_scenario.sort_values(by='loss_tot', ascending=False)

            fig_name = 'fig_SC_' + sc_haz_str + '_loss_sys_vs_comptype_s1.png'
            draw_component_loss_barchart_s1(ctype_resp_sorted, scenario_tag, hazards.hazard_type, config.OUTPUT_PATH, fig_name)

            fig_name = 'fig_SC_' + sc_haz_str + '_loss_sys_vs_comptype_s2.png'
            draw_component_loss_barchart_s2(ctype_resp_sorted, scenario_tag, hazards.hazard_type, config.OUTPUT_PATH, fig_name)

            fig_name = 'fig_SC_' + sc_haz_str + '_loss_sys_vs_comptype_s3.png'
            draw_component_loss_barchart_s3(ctype_resp_sorted, scenario_tag, hazards.hazard_type, config.OUTPUT_PATH, fig_name)

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # FAILURE RATE -- PERCENTAGE of component types

            fig_name = 'fig_SC_' + sc_haz_str + '_comptype_failures.png'
            draw_component_failure_barchart(uncosted_comptypes, ctype_failure_mean, scenario_header, scenario_tag, hazards.hazard_type, config.OUTPUT_PATH, fig_name)

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # RESTORATION PROGNOSIS for specified scenarios

            component_fullrst_time, ctype_scenario_outcomes = calc_comptype_damage_scenario_given_hazard(infrastructure, scenario, hazards, ctype_resp_sorted, component_response, cp_types_costed, scenario_header)

            # All the nodes that need to be fixed for each output node:
            repair_list_combined = prep_repair_list(infrastructure, component_meanloss, component_fullrst_time, comps_uncosted, weight_criteria, scenario_header)

            repair_path = copy.deepcopy(repair_list_combined)
            output_node_list = infrastructure.output_nodes.keys()

            for num_rst_steams in RESTORATION_STREAMS:
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # SYSTEM RESTORATION for given scenario & restoration setup
                rst_setup_filename = 'restoration_setup_' + sc_haz_str + '_' + hazards.intensity_measure_unit + '_' + hazards.intensity_measure_param + '.csv'

                rst_setup_df = calc_restoration_setup(component_meanloss, output_node_list, comps_uncosted, repair_list_combined, num_rst_steams, RESTORATION_OFFSET, component_fullrst_time, scenario.output_path, scenario_header, rst_setup_filename)

                # Check if nothing to repair, i.e. if repair list is empty:
                comps_to_repair = rst_setup_df.index.values.tolist()
                if not comps_to_repair:
                    rootLogger.info("*** Scenario: " + scenario_tag)
                    rootLogger.info("Nothing to repair. Time to repair is zero.")
                    rootLogger.info("Skipping repair visualisation for this scenario.")
                    break

                fig_rst_gantt_name = 'fig_SC_' + sc_haz_str + '_str' + str(num_rst_steams) + '_restoration.png'
                restoration_timeline_array, time_to_full_restoration_for_lines = \
                    vis_restoration_process(scenario, infrastructure, rst_setup_df, num_rst_steams, repair_path, fig_rst_gantt_name, scenario_tag, hazards.hazard_type)

                time_to_full_restoration_for_lines_df[(sc_haz_str, num_rst_steams)] = \
                    [time_to_full_restoration_for_lines[x] for x in output_node_list]

                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # COMPONENT CRITICALITY for given scenario & restoration setup
                fig_name = 'fig_SC_' + sc_haz_str + '_str' + str(num_rst_steams) + '_component_criticality.png'
                component_criticality(infrastructure, scenario, ctype_scenario_outcomes, hazards.hazard_type, scenario_tag, fig_name)

        # --------------------------------------------------------------------------
        # *** END : FOCAL_HAZARD_SCENARIOS FOR LOOP ***
        time_to_full_restoration_for_lines_csv = os.path.join(config.OUTPUT_PATH, 'line_restoration_prognosis.csv')
        time_to_full_restoration_for_lines_df.to_csv(time_to_full_restoration_for_lines_csv, sep=',')



        rootLogger.info('End: SCENARIO LOSS ANALYSIS')

        # ---------------------------------------------------------------------

    else:
        print("Input file not found: " + str(args.setup))

    rootLogger.info('End')


if __name__ == "__main__":
    main()
