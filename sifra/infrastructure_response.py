import os
import sys
import time
from datetime import timedelta
import cPickle
import zipfile

import numpy as np
import pandas as pd
import parmap

from model_ingest import ingest_spreadsheet
from sifraclasses import Scenario
from sifra.modelling.hazard_levels import HazardLevels


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from colorama import Fore

def run_scenario(config_file):
    scenario = Scenario(config_file)

    infrastructure = ingest_spreadsheet(config_file)
    hazard_levels = HazardLevels(scenario)

    # Use the parallel option in the scenario to determine how
    # to run
    response_array = []
    response_array.extend(parmap.map(infrastructure.expose_to,
                                     hazard_levels.hazard_range(),
                                     scenario,
                                     parallel=False))

    # combine the responses into one dict
    response_dict = {}
    for response in response_array:
        response_dict.update(response)


    post_processing(infrastructure, scenario, response_dict)

# ****************************************************************************
# BEGIN POST-PROCESSING ...
# ****************************************************************************

def plot_mean_econ_loss(fc, sc, economic_loss_array):
    """Draws and saves a boxplot of mean economic loss"""

    fig = plt.figure(figsize=(9, 5), facecolor='white')
    sns.set(style='ticks', palette='Set3')
    # ax = sns.boxplot(economic_loss_array*100, showmeans=True,
    #                  widths=0.3, linewidth=0.7, color='lightgrey',
    #                  meanprops=dict(marker='s',
    #                                 markeredgecolor='salmon',
    #                                 markerfacecolor='salmon')
    #                 )
    ax = sns.boxplot(economic_loss_array * 100, showmeans=True,
                     linewidth=0.7, color='lightgrey',
                     meanprops=dict(marker='s',
                                    markeredgecolor='salmon',
                                    markerfacecolor='salmon')
                     )
    sns.despine(top=True, left=True, right=True)
    ax.tick_params(axis='y', left='off', right='off')
    ax.yaxis.grid(True)

    intensity_label = sc.intensity_measure_param+' ('\
                      +sc.intensity_measure_unit+')'
    ax.set_xlabel(intensity_label)
    ax.set_ylabel('Loss Fraction (%)')
    ax.set_xticklabels(sc.hazard_intensity_vals);
    ax.set_title('Loss Ratio', loc='center', y=1.04);
    ax.title.set_fontsize(12)

    figfile = os.path.join(sc.output_path, 'fig_lossratio_boxplot.png')
    plt.savefig(figfile, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)


def post_processing(infrastructure, scenario, response_dict):
    # ------------------------------------------------------------------------
    # 'ids_comp_vs_haz' is a dict of numpy arrays
    # We pickle it for archival. But the file size can get very large.
    # So we zip it for archival and delete the original
    idshaz = os.path.join(scenario.raw_output_dir, 'ids_comp_vs_haz.pickle')
    with open(idshaz, 'w') as handle:
        for response_key in sorted(response_dict.iterkeys()):
            cPickle.dump({response_key: response_dict[response_key][0]}, handle)

    crd_pkl = os.path.join(scenario.raw_output_dir, 'component_resp_dict.pickle')
    with open(crd_pkl, 'w') as handle:
        for response_key in sorted(response_dict.iterkeys()):
            cPickle.dump({response_key: response_dict[response_key][2]}, handle)

    sod_pkl = os.path.join(scenario.raw_output_dir, 'sys_output_dict.pickle')
    with open(sod_pkl, 'w') as handle:
        for response_key in sorted(response_dict.iterkeys()):
            cPickle.dump({response_key: response_dict[response_key][1]}, handle)

    idshaz_zip = os.path.join(scenario.raw_output_dir, 'ids_comp_vs_haz.zip')
    zipmode = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(idshaz_zip, 'w', zipmode) as zip:
        zip.write(idshaz)
    os.remove(idshaz)

    # ------------------------------------------------------------------------
    # System output file (for given hazard transfer parameter value)
    # ------------------------------------------------------------------------
    sys_output_dict= dict()
    for response_key in sorted(response_dict.iterkeys()):
        sys_output_dict[response_key] = response_dict[response_key][1]

    sys_output_df = pd.DataFrame(sys_output_dict)
    sys_output_df.index.name = 'Output Nodes'

    outfile_sysoutput = os.path.join(scenario.output_path,
                                 'system_output_given_haz_param.csv')
    sys_output_df.to_csv(outfile_sysoutput,
                         sep=',', index_label=['Output Nodes'])

    # ------------------------------------------------------------------------
    # Loss calculations by Component Type
    # ------------------------------------------------------------------------
    tp_ct = []
    for comp_type in infrastructure.get_component_types():
        tp_ct.extend(
            ((comp_type, 'loss_mean'), (comp_type, 'loss_std'),
             (comp_type, 'loss_tot'), (comp_type, 'func_mean'),
             (comp_type, 'func_std'))
        )

    mindex = pd.MultiIndex.from_tuples(tp_ct,
                                       names=['component_type', 'response'])
    comptype_resp_df = pd.DataFrame(index=mindex,
                                    columns=[scenario.hazard_intensity_vals])
    comptype_resp_dict = comptype_resp_df.to_dict()

    for p in scenario.hazard_intensity_vals:
        for component_type in infrastructure.get_component_types():
            components_of_type = list(infrastructure.get_components_for_type(component_type))
            ct_loss_mean_list = [response_dict[p][2][(comp_id, 'loss_mean')] for comp_id in components_of_type]
            comptype_resp_dict[p][(component_type, 'loss_mean')] = np.mean(ct_loss_mean_list)

            ct_loss_mean_list = [response_dict[p][2][(comp_id, 'loss_mean')] for comp_id in components_of_type]
            comptype_resp_dict[p][(component_type, 'loss_tot')] = np.sum(ct_loss_mean_list)

            ct_loss_std_list = [response_dict[p][2][(comp_id, 'loss_std')] for comp_id in components_of_type]
            comptype_resp_dict[p][(component_type, 'loss_std')] = np.mean(ct_loss_std_list)

            ct_func_mean_list = [response_dict[p][2][(comp_id, 'func_mean')] for comp_id in components_of_type]
            comptype_resp_dict[p][(component_type, 'func_mean')] = np.mean(ct_func_mean_list)

            ct_func_std_list = [response_dict[p][2][(comp_id, 'func_std')] for comp_id in components_of_type]
            comptype_resp_dict[p][(component_type, 'func_std')] = np.mean(ct_func_std_list)

            ct_num_failures_list = [response_dict[p][2][(comp_id, 'num_failures')] for comp_id in components_of_type]
            comptype_resp_dict[p][(component_type, 'num_failures')] = np.mean(ct_num_failures_list)

    # ------------------------------------------------------------------------
    # Calculating system fragility:
    sys_frag = np.zeros((scenario.num_samples, scenario.num_hazard_pts), dtype=int)
    if_system_damage_states = infrastructure.get_dmg_scale_bounds(scenario)
    for j, hazard_level in enumerate(scenario.hazard_intensity_vals):
        for i in range(scenario.num_samples):
            # system output and economic loss
            sys_frag[i, j] = np.sum(response_dict[hazard_level][4][j] > if_system_damage_states)

    # Calculating Probability of Exceedence:
    pe_sys_econloss = np.zeros((len(infrastructure.get_system_damage_states()), scenario.num_hazard_pts))
    for j in range(scenario.num_hazard_pts):
        for i in range(len(infrastructure.get_system_damage_states())):
            pe_sys_econloss[i, j] = \
                np.sum(sys_frag[:, j] >= i) / float(scenario.num_samples)

    # ------------------------------------------------------------------------
    # For Probability of Exceedence calculations based on component failures
    # ------------------------------------------------------------------------
    #
    #   Damage state boundaries for Component Type Failures (Substations) are
    #   based on HAZUS MH MR3, p 8-66 to 8-68
    #
    # ------------------------------------------------------------------------

    cp_classes_in_system = np.unique(list(infrastructure.get_component_class_list()))

    cp_class_map = {k: [] for k in cp_classes_in_system}
    for k, v in fc.compdict['component_class'].iteritems():
        cp_class_map[v].append(k)

    # ------------------------------------------------------------------------

    if fc.system_class == 'Substation':
        uncosted_classes = ['JUNCTION POINT',
                            'SYSTEM INPUT', 'SYSTEM OUTPUT',
                            'Generator', 'Bus', 'Lightning Arrester']
        ds_lims_compclasses = {
            'Disconnect Switch': [0.05, 0.40, 0.70, 0.99, 1.00],
            'Circuit Breaker': [0.05, 0.40, 0.70, 0.99, 1.00],
            'Current Transformer': [0.05, 0.40, 0.70, 0.99, 1.00],
            'Voltage Transformer': [0.05, 0.40, 0.70, 0.99, 1.00],
            'Power Transformer': [0.05, 0.40, 0.70, 0.99, 1.00],
            'Control Building': [0.06, 0.30, 0.75, 0.99, 1.00]
        }

        cp_classes_costed = \
            [x for x in cp_classes_in_system if x not in uncosted_classes]

        # --- System fragility - Based on Failure of Component Classes ---
        comp_class_failures = \
            {cc: np.zeros((scenario.num_samples, scenario.num_hazard_pts))
             for cc in cp_classes_costed}

        comp_class_frag = {cc: np.zeros((scenario.num_samples, scenario.num_hazard_pts))
                           for cc in cp_classes_costed}
        for j, PGA in enumerate(scenario.hazard_intensity_str):
            for i in range(scenario.num_samples):
                for compclass in cp_classes_costed:
                    for c in cp_class_map[compclass]:
                        comp_class_failures[compclass][i, j] += \
                            ids_comp_vs_haz[PGA][
                                i, fc.network.node_map[c]
                            ]
                    comp_class_failures[compclass][i, j] /= \
                        len(cp_class_map[compclass])

                    comp_class_frag[compclass][i, j] = \
                        np.sum(comp_class_failures[compclass][i, j]
                               > ds_lims_compclasses[compclass])

        # Probability of Exceedence -- Based on Failure of Component Classes
        pe_sys_cpfailrate = np.zeros(
            (len(fc.sys_dmg_states), scenario.num_hazard_pts)
        )
        for p in range(scenario.num_hazard_pts):
            for d in range(len(fc.sys_dmg_states)):
                ds_ss_ix = []
                for compclass in cp_classes_costed:
                    ds_ss_ix.append(
                        np.sum(comp_class_frag[compclass][:, p] >= d) /
                        float(scenario.num_samples)
                    )
                pe_sys_cpfailrate[d, p] = np.median(ds_ss_ix)

        # --- Save prob exceedance data as npy ---
        np.save(os.path.join(scenario.raw_output_dir, 'pe_sys_cpfailrate.npy'),
                pe_sys_cpfailrate)

    # ------------------------------------------------------------------------

    if fc.system_class == 'PowerStation':
        uncosted_classes = ['JUNCTION POINT', 'SYSTEM INPUT', 'SYSTEM OUTPUT']
        ds_lims_compclasses = {
            'Boiler': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Control Building': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Emission Management': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Fuel Delivery and Storage': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Fuel Movement': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Generator': [0.0, 0.05, 0.40, 0.70, 1.00],
            'SYSTEM OUTPUT': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Stepup Transformer': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Turbine': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Water System': [0.0, 0.05, 0.40, 0.70, 1.00]
        }

    # ------------------------------------------------------------------------
    # Validate damage ratio of the system
    # ------------------------------------------------------------------------

    exp_damage_ratio = np.zeros((len(fc.network.nodes_all),
                                 scenario.num_hazard_pts))
    for l, PGA in enumerate(scenario.hazard_intensity_vals):
        # compute expected damage ratio
        for j, comp_name in enumerate(fc.network.nodes_all):
            pb = pe2pb(
                cal_pe_ds(comp_name, PGA, fc.compdict, fc.fragdict, sc)
            )
            comp_type = fc.compdict['component_type'][comp_name]
            dr = np.array([fc.fragdict['damage_ratio'][comp_type][ds]
                           for ds in fc.sys_dmg_states])
            cf = fc.compdict['cost_fraction'][comp_name]
            loss_list = dr * cf
            exp_damage_ratio[j, l] = np.sum(pb * loss_list)

    # ------------------------------------------------------------------------
    # Time to Restoration of Full Capacity
    # ------------------------------------------------------------------------

    threshold = 0.99
    required_time = []

    for j in range(scenario.num_hazard_pts):
        cpower = \
            np.mean(output_array_given_recovery[:, j, :], axis=0) / \
            fc.nominal_production
        temp = cpower > threshold
        if sum(temp) > 0:
            required_time.append(np.min(scenario.restoration_time_range[temp]))
        else:
            required_time.append(scenario.restore_time_max)

    # ------------------------------------------------------------------------
    # Write analytical outputs to file
    # ------------------------------------------------------------------------

    # --- Output File --- summary output ---
    outfile_sys_response = os.path.join(
        scenario.output_path, 'system_response.csv')
    out_cols = ['PGA',
                'Economic Loss',
                'Mean Output',
                'Days to Full Recovery']
    outdat = {out_cols[0]: scenario.hazard_intensity_vals,
              out_cols[1]: np.mean(economic_loss_array, axis=0),
              out_cols[2]: np.mean(calculated_output_array, axis=0),
              out_cols[3]: required_time}
    df = pd.DataFrame(outdat)
    df.to_csv(
        outfile_sys_response, sep=',',
        index=False, columns=out_cols
    )

    # --- Output File --- response of each COMPONENT to hazard ---
    outfile_comp_resp = os.path.join(
        scenario.output_path, 'component_response.csv')
    component_resp_df = pd.DataFrame(component_resp_dict)
    component_resp_df.index.names = ['component_id', 'response']
    component_resp_df.to_csv(
        outfile_comp_resp, sep=',',
        index_label=['component_id', 'response']
    )

    # --- Output File --- mean loss of component ---
    outfile_comp_loss = os.path.join(
        scenario.output_path, 'component_meanloss.csv')
    component_loss_df = component_resp_df.iloc[
        component_resp_df.index.get_level_values(1) == 'loss_mean']
    component_loss_df.reset_index(level='response', inplace=True)
    component_loss_df = component_loss_df.drop('response', axis=1)
    component_loss_df.to_csv(
        outfile_comp_loss, sep=',',
        index_label=['component_id']
    )

    # --- Output File --- response of each COMPONENT TYPE to hazard ---
    outfile_comptype_resp = os.path.join(
        scenario.output_path, 'comp_type_response.csv')
    comptype_resp_df = pd.DataFrame(comptype_resp_dict)
    comptype_resp_df.index.names = ['component_type', 'response']
    comptype_resp_df.to_csv(
        outfile_comptype_resp, sep=',',
        index_label=['component_type', 'response']
    )

    # --- Output File --- mean loss of component type ---
    outfile_comptype_loss = os.path.join(
        scenario.output_path, 'comp_type_meanloss.csv')
    comptype_loss_df = comptype_resp_df.iloc[
        comptype_resp_df.index.get_level_values(1) == 'loss_mean']
    comptype_loss_df.reset_index(level='response', inplace=True)
    comptype_loss_df = comptype_loss_df.drop('response', axis=1)
    comptype_loss_df.to_csv(
        outfile_comptype_loss, sep=',',
        index_label=['component_type']
    )

    # --- Output File --- mean failures for component types ---
    outfile_comptype_failures = os.path.join(
        scenario.output_path, 'comp_type_meanfailures.csv')
    comptype_failure_df = comptype_resp_df.iloc[
        comptype_resp_df.index.get_level_values(1) == 'num_failures']
    comptype_failure_df.reset_index(level='response', inplace=True)
    comptype_failure_df = comptype_failure_df.drop('response', axis=1)
    comptype_failure_df.to_csv(
        outfile_comptype_failures, sep=',',
        index_label=['component_type']
    )

    # # --- Output File --- DataFrame of mean failures per component CLASS ---
    # outfile_compclass_failures = os.path.join(
    #     output_path, 'comp_class_meanfailures.csv')
    # compclass_failure_df.to_csv(outfile_compclass_failures, sep=',',
    #                         index_label=['component_class'])

    # ------------------------------------------------------------------------
    # *** Saving vars ***
    # ------------------------------------------------------------------------

    if scenario.save_vars_npy:
        np.save(
            os.path.join(scenario.raw_output_dir, 'economic_loss_array.npy'),
            economic_loss_array
        )

        np.save(
            os.path.join(scenario.raw_output_dir, 'calculated_output_array.npy'),
            calculated_output_array
        )

        np.save(
            os.path.join(scenario.raw_output_dir,
                         'output_array_given_recovery.npy'),
            output_array_given_recovery
        )

        np.save(
            os.path.join(scenario.raw_output_dir, 'exp_damage_ratio.npy'),
            exp_damage_ratio
        )

        np.save(
            os.path.join(scenario.raw_output_dir, 'sys_frag.npy'),
            sys_frag
        )

        np.save(
            os.path.join(scenario.raw_output_dir, 'required_time.npy'),
            required_time
        )

        np.save(
            os.path.join(scenario.raw_output_dir, 'pe_sys_econloss.npy'),
            pe_sys_econloss
        )
    # ------------------------------------------------------------------------
    print("\nOutputs saved in: " +
          Fore.GREEN + scenario.output_path + Fore.RESET + '\n')

    plot_mean_econ_loss(fc, sc, economic_loss_array)

    # ... END POST-PROCESSING
    # ****************************************************************************


def main():
    code_start_time = time.time()

    SETUPFILE = sys.argv[1]

    run_scenario(SETUPFILE)
    print("[ Run time: %s ]\n" % \
          str(timedelta(seconds=(time.time() - code_start_time))))


if __name__ == '__main__':
    main()

