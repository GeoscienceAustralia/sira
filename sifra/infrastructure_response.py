from __future__ import print_function
import os
import time
from datetime import timedelta
import pickle
import zipfile

import numpy as np
import pandas as pd
import parmap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sifra.modelling.hazard import Hazards
from sifra.logger import rootLogger
from sifra.configuration import Configuration


def run_para_scen(hazard_level, infrastructure, scenario):
    """
    The parmap.map function requires a module level function as a parameter.
    So this function satisfies that requirement by calling the infrastructure's
    exponse_to method within this one.
    :param hazard_level: The hazard level that the infrastructure will be exposed to
    :param infrastructure: The infrastructure model that is being simulated
    :param scenario: The Parameters for the simulation
    :return: List of results of the simulation
    """

    return infrastructure.expose_to(hazard_level, scenario)


# ****************************************************************************
# BEGIN POST-PROCESSING ...
# ****************************************************************************


def plot_mean_econ_loss(scenario, economic_loss_array, hazard):
    """Draws and saves a boxplot of mean economic loss"""
    hazvals_ext = [[str(i)] * scenario.num_samples for i in hazard.hazard_scenario_list]

    x1 = np.ndarray.flatten(np.array(hazvals_ext))

    smpl = range(1, scenario.num_samples + 1, 1)
    x2 = np.array(smpl * hazard.num_hazard_pts)

    arrays = [x1, x2]
    econ_loss = np.array(economic_loss_array)
    econ_loss = np.ndarray.flatten(econ_loss.transpose())
    econ_loss_flat = np.ndarray.flatten(econ_loss)

    econ_loss_df = pd.DataFrame(econ_loss_flat, index=arrays)
    econ_loss_df.index.names = ['Hazard Intensity', 'Sample Num']
    econ_loss_df.columns = ['Econ Loss Ratio']

    fig = plt.figure(figsize=(9, 5), facecolor='white')
    sns.set(style='ticks', palette='Set2')
    # whitesmoke='#F5F5F5', coral='#FF7F50'
    ax = sns.boxplot(x=x1, y='Econ Loss Ratio', data=econ_loss_df,
                     linewidth=0.8, color='whitesmoke',
                     showmeans=True,
                     meanprops=dict(marker='o',
                                    markeredgecolor='coral',
                                    markerfacecolor='coral')
                     )

    sns.despine(bottom=False, top=True, left=True, right=True, offset=10)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['bottom'].set_color('#555555')

    ax.yaxis.grid(True, which="major", linestyle='-',
                  linewidth=0.4, color='#B6B6B6')

    ax.tick_params(axis='x', bottom='on', top='off',
                   width=0.8, labelsize=8, pad=5, color='#555555')
    ax.tick_params(axis='y', left='off', right='off',
                   width=0.8, labelsize=8, pad=5, color='#555555')

    ax.set_xticklabels(hazard.hazard_scenario_list)
    intensity_label \
        = hazard.intensity_measure_param+' ('+hazard.intensity_measure_unit+')'
    ax.set_xlabel(intensity_label, labelpad=9, size=10)
    ax.set_ylabel('Loss Fraction (%)', labelpad=9, size=10)

    ax.set_title('Loss Ratio', loc='center', y=1.04)
    ax.title.set_fontsize(12)

    figfile = os.path.join(scenario.output_path, 'fig_lossratio_boxplot.png')
    plt.savefig(figfile, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)



def write_system_response(response_list, scenario):

    # ------------------------------------------------------------------------
    # 'ids_comp_vs_haz' is a dict of numpy arrays
    # We pickle it for archival. But the file size can get very large.
    # So we zip it for archival and delete the original
    # ------------------------------------------------------------------------
    idshaz = os.path.join(scenario.raw_output_dir, 'ids_comp_vs_haz.pickle')
    id_comp_vs_haz = response_list[0]
    with open(idshaz, 'w') as handle:
        for response_key in sorted(id_comp_vs_haz.keys()):
            pickle.dump({response_key: id_comp_vs_haz[response_key]}, handle)
    idshaz_zip = os.path.join(scenario.raw_output_dir, 'ids_comp_vs_haz.zip')
    zipmode = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(idshaz_zip, 'w', zipmode) as zip:
        zip.write(idshaz)
    os.remove(idshaz)

    # ------------------------------------------------------------------------
    # System output file (for given hazard transfer parameter value)
    # ------------------------------------------------------------------------
    sys_output_dict = response_list[1]
    sod_pkl = os.path.join(scenario.raw_output_dir,
                           'sys_output_dict.pickle')
    with open(sod_pkl, 'w') as handle:
        for response_key in sorted(sys_output_dict.keys()):
            pickle.dump({response_key: sys_output_dict[response_key]},
                        handle)

    sys_output_df = pd.DataFrame(sys_output_dict)
    sys_output_df = sys_output_df.transpose()
    sys_output_df.index.name = 'Hazard Intensity'

    outfile_sysoutput = os.path.join(scenario.output_path,
                                     'system_output_vs_haz_intensity.csv')
    sys_output_df.to_csv(outfile_sysoutput,
                         sep=',',
                         index_label=[sys_output_df.index.name])

    # ------------------------------------------------------------------------
    # Hazard response for component instances, i.e. components as-installed
    # ------------------------------------------------------------------------
    component_resp_dict = response_list[2]
    crd_pkl = os.path.join(scenario.raw_output_dir,
                           'component_resp_dict.pickle')
    with open(crd_pkl, 'w') as handle:
        for response_key in sorted(component_resp_dict.keys()):
            pickle.dump({response_key: component_resp_dict[response_key]},
                        handle)


def loss_by_comp_type(response_list, infrastructure, scenario, hazard):
    """
    Aggregate the economic loss statistics by component type.
    :param response_list: list of simulation results
    :param infrastructure: simulated infrastructure
    :param scenario: values used in simulation
    :return: None
    """
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



    mindex = pd.MultiIndex.from_tuples(
        tp_ct,
        names=['component_type', 'response'])
    comptype_resp_df = pd.DataFrame(
        index=mindex,
        columns=[hazard.hazard_scenario_list])
    comptype_resp_dict = {}

    component_resp_dict = response_list[2]
    for p in hazard.hazard_scenario_list:
        if p not in comptype_resp_dict:
            comptype_resp_dict[p] = dict()

        for component_type in infrastructure.get_component_types():

            components_of_type \
                = list(infrastructure.get_components_for_type(component_type))

            ct_loss_mean_list \
                = [component_resp_dict[p][(comp_id, 'loss_mean')]
                   for comp_id in components_of_type]

            comptype_resp_dict[p][(component_type, 'loss_mean')] \
                = np.mean(ct_loss_mean_list)

            ct_loss_mean_list \
                = [component_resp_dict[p][(comp_id, 'loss_mean')]
                   for comp_id in components_of_type]

            comptype_resp_dict[p][(component_type, 'loss_tot')] \
                = np.sum(ct_loss_mean_list)

            ct_loss_std_list \
                = [component_resp_dict[p][(comp_id, 'loss_std')]
                   for comp_id in components_of_type]

            comptype_resp_dict[p][(component_type, 'loss_std')] \
                = np.mean(ct_loss_std_list)

            ct_func_mean_list \
                = [component_resp_dict[p][(comp_id, 'func_mean')]
                   for comp_id in components_of_type]

            comptype_resp_dict[p][(component_type, 'func_mean')] \
                = np.mean(ct_func_mean_list)

            ct_func_std_list \
                = [component_resp_dict[p][(comp_id, 'func_std')]
                   for comp_id in components_of_type]

            comptype_resp_dict[p][(component_type, 'func_std')] \
                = np.mean(ct_func_std_list)

            ct_num_failures_list \
                = [component_resp_dict[p][(comp_id, 'num_failures')]
                   for comp_id in components_of_type]

            comptype_resp_dict[p][(component_type, 'num_failures')] \
                = np.mean(ct_num_failures_list)

    # ------------------------------------------------------------------------
    # Calculating system fragility:
    economic_loss_array = response_list[4]
    sys_frag = np.zeros_like(economic_loss_array, dtype=int)
    if_system_damage_states = infrastructure.get_dmg_scale_bounds(scenario)
    for j, hazard_level in enumerate(hazard.hazard_scenario_list):
        for i in range(scenario.num_samples):
            # system output and economic loss
            sys_frag[i, j] = \
                np.sum(economic_loss_array[i, j] > if_system_damage_states)

    # Calculating Probability of Exceedence:
    pe_sys_econloss = np.zeros(
        (len(infrastructure.get_system_damage_states()),
         hazard.num_hazard_pts)
    )
    for j in range(hazard.num_hazard_pts):
        for i in range(len(infrastructure.get_system_damage_states())):
            pe_sys_econloss[i, j] = \
                np.sum(sys_frag[:, j] >= i) / float(scenario.num_samples)

    # --- Output File --- response of each COMPONENT TYPE to hazard ---
    outfile_comptype_resp = os.path.join(
        scenario.output_path, 'comptype_response.csv')
    comptype_resp_df = pd.DataFrame(comptype_resp_dict)
    comptype_resp_df.index.names = ['component_type', 'response']
    comptype_resp_df.to_csv(
        outfile_comptype_resp, sep=',',
        index_label=['component_type', 'response']
    )

    # --- Output File --- mean loss of component type ---
    outfile_comptype_loss = os.path.join(
        scenario.output_path, 'comptype_meanloss.csv')
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
        scenario.output_path, 'comptype_meanfailures.csv')
    comptype_failure_df = comptype_resp_df.iloc[
        comptype_resp_df.index.get_level_values(1) == 'num_failures']
    comptype_failure_df.reset_index(level='response', inplace=True)
    comptype_failure_df = comptype_failure_df.drop('response', axis=1)
    comptype_failure_df.to_csv(
        outfile_comptype_failures, sep=',',
        index_label=['component_type']
    )

    np.save(
        os.path.join(scenario.raw_output_dir, 'sys_frag.npy'),
        sys_frag
    )

    np.save(
        os.path.join(scenario.raw_output_dir, 'pe_sys_econloss.npy'),
        pe_sys_econloss
    )


def pe_by_component_class(response_list, infrastructure, scenario, hazard):
    """
    Calculated  probability of exceedence based on component classes
    :param response_list:
    :param infrastructure:
    :param scenario:
    :param hazard:
    :return:
    """
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
    for comp_id, component in infrastructure.components.items():
        cp_class_map[component.component_class].append(component)

    # ------------------------------------------------------------------------
    # For Probability of Exceedence calculations based on component failures:
    #   Damage state boundaries for Component Type Failures (Substations) are
    #   based on HAZUS MH MR3, p 8-66 to 8-68
    # ------------------------------------------------------------------------
    if infrastructure.system_class == 'Substation':
        cp_classes_costed = \
            [x for x in cp_classes_in_system
             if x not in infrastructure.uncosted_classes]

        # --- System fragility - Based on Failure of Component Classes ---
        comp_class_failures = \
            {cc: np.zeros((scenario.num_samples, scenario.num_hazard_pts))
             for cc in cp_classes_costed}

        comp_class_frag = \
            {cc: np.zeros((scenario.num_samples, scenario.num_hazard_pts))
             for cc in cp_classes_costed}

        # TODO check or correctness
        # for j, hazard_level in enumerate(hazard.hazard_range):
        #     for i in range(scenario.num_samples):
        #         for compclass in cp_classes_costed:
        #             for c in cp_class_map[compclass]:
        #                 comp_class_failures[compclass][i, j] += \
        #                     response_list[hazard_level.hazard_intensity]\
        #                                  [i, infrastructure.components[c]]
        #             comp_class_failures[compclass][i, j] /= \
        #                 len(cp_class_map[compclass])
        #
        #             comp_class_frag[compclass][i, j] = \
        #                 np.sum(comp_class_failures[compclass][i, j] > \
        #                        infrastructure.ds_lims_compclasses[compclass])

        for j, hazard_intensity in enumerate(hazard.hazard_range):
            for i in range(scenario.num_samples):
                for compclass in cp_classes_costed:
                    for c in cp_class_map[compclass]:
                        comp_class_failures[compclass][i, j] += \
                            response_list[hazard_intensity]\
                                         [i, infrastructure.components[c]]
                    comp_class_failures[compclass][i, j] /= \
                        len(cp_class_map[compclass])

                    comp_class_frag[compclass][i, j] = \
                        np.sum(comp_class_failures[compclass][i, j] > \
                               infrastructure.ds_lims_compclasses[compclass])

        # Probability of Exceedence -- Based on Failure of Component Classes
        pe_sys_cpfailrate = np.zeros(
            (len(infrastructure.sys_dmg_states), scenario.num_hazard_pts)
        )
        for p in range(scenario.num_hazard_pts):
            for d in range(len(infrastructure.sys_dmg_states)):
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
    # Validate damage ratio of the system
    # ------------------------------------------------------------------------

    exp_damage_ratio = np.zeros((len(infrastructure.components),
                                 hazard.num_hazard_pts))
    for l, hazard_scenario_name in enumerate(hazard.hazard_scenario_list):
        # compute expected damage ratio
        for j, component in enumerate(infrastructure.components.values()):
            # TODO remove invalid Component accesses !!

            component_pe_ds = np.zeros(len(component.damage_states))
            for damage_state_index in component.damage_states.keys():
                long, lat = component.get_location()
                hazard_intensity = hazard.get_hazard_intensity_at_location(hazard_scenario_name, long, lat)
                component_pe_ds[damage_state_index] = component.damage_states[damage_state_index].response_function(
                    hazard_intensity)

            component_pe_ds = component_pe_ds[1:]

            pb = pe2pb(component_pe_ds)

            dr = np.array([component.damage_states[int(ds)].damage_ratio
                           for ds in infrastructure.sys_dmg_states])
            cf = component.cost_fraction
            loss_list = dr * cf
            exp_damage_ratio[j, l] = np.sum(pb * loss_list)

    # ------------------------------------------------------------------------
    # Write analytical outputs to file
    # ------------------------------------------------------------------------

    # --- Output File --- summary output ---
    outfile_sys_response = os.path.join(
        scenario.output_path, 'system_response.csv')
    out_cols = ['INTENSITY_MEASURE',
                'Economic Loss',
                'Mean Output']

    # create the arrays
    comp_response_list = response_list[2]
    economic_loss_array = response_list[4]
    calculated_output_array = response_list[3]

    outdat = {out_cols[0]: hazard.hazard_scenario_list,
              out_cols[1]: np.mean(economic_loss_array, axis=0),
              out_cols[2]: np.mean(calculated_output_array, axis=0)}
    df = pd.DataFrame(outdat)
    df.to_csv(
        outfile_sys_response, sep=',',
        index=False, columns=out_cols
    )

    # --- Output File --- response of each COMPONENT to hazard ---
    outfile_comp_resp = os.path.join(scenario.output_path,
                                     'component_response.csv')
    component_resp_df = pd.DataFrame(comp_response_list)
    component_resp_df.index.names = ['component_id', 'response']
    component_resp_df.columns = hazard.hazard_scenario_list
    component_resp_df.to_csv(
        outfile_comp_resp, sep=',',
        index_label=['component_id', 'response']
    )

    # --- Output File --- mean loss of component ---
    outfile_comp_loss = os.path.join(scenario.output_path,
                                     'component_meanloss.csv')
    component_loss_df = component_resp_df.iloc\
        [component_resp_df.index.get_level_values(1) == 'loss_mean']
    component_loss_df.reset_index(level='response', inplace=True)
    component_loss_df = component_loss_df.drop('response', axis=1)
    component_loss_df.to_csv(
        outfile_comp_loss, sep=',',
        index_label=['component_id']
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
            os.path.join(scenario.raw_output_dir, 'exp_damage_ratio.npy'),
            exp_damage_ratio
        )

    # ------------------------------------------------------------------------
        rootLogger.info("Outputs saved in: " + scenario.output_path)

    # ... END POST-PROCESSING
    # ****************************************************************************

def pe2pb(pe):
    """
    Convert probability of excedence of damage states, to
    probability of being in each discrete damage state
    """
    # sorted array: from max to min
    pex = np.sort(pe)[::-1]
    tmp = -1.0 * np.diff(pex)
    pb = np.append(tmp, pex[-1])
    pb = np.insert(pb, 0, 1 - pex[0])
    return pb