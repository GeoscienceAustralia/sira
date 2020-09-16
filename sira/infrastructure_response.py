from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range

import os
import pickle
import zipfile

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns

import logging
rootLogger = logging.getLogger(__name__)


# ****************************************************************************
# BEGIN POST-PROCESSING ...
# ****************************************************************************

def calc_tick_vals(val_list, xstep=0.1):
    num_ticks = int(round(len(val_list)/xstep)) + 1
    if num_ticks>12 and num_ticks<=20:
        xstep = 0.2
        num_ticks = int(round(len(val_list)/xstep)) + 1
    elif num_ticks>20:
        num_ticks = 11
    tick_labels = val_list[::(num_ticks-1)]
    if type(tick_labels[0])==float:
        tick_labels = ['{:.3f}'.format(val) for val in tick_labels]
    return tick_labels


def plot_mean_econ_loss(scenario, economic_loss_array, hazards):
    """Draws and saves a boxplot of mean economic loss"""
    hazvals_ext = [[str(i)] * scenario.num_samples
                   for i in hazards.hazard_scenario_list]

    x1 = np.ndarray.flatten(np.array(hazvals_ext))

    smpl = list(range(1, scenario.num_samples + 1, 1))
    x2 = np.array(smpl * hazards.num_hazard_pts)

    arrays = [x1, x2]
    econ_loss = np.array(economic_loss_array)
    econ_loss = np.ndarray.flatten(econ_loss.transpose())
    econ_loss_flat = np.ndarray.flatten(econ_loss)

    econ_loss_df = pd.DataFrame(econ_loss_flat, index=arrays)
    econ_loss_df.index.names = ['Hazard Intensity', 'Sample Num']
    econ_loss_df.columns = ['Econ Loss Ratio']

    fig = plt.figure(figsize=(9, 5), facecolor='white')
    sns.set(style='ticks', palette='Set2')
    ax = sns.boxplot(x=x1, y='Econ Loss Ratio',
                     data=econ_loss_df,
                     linewidth=0.8, color='whitesmoke',
                     showmeans=True,
                     showfliers=True,
                     meanprops=dict(marker='o',
                                    markeredgecolor='coral',
                                    markerfacecolor='coral')
                     )

    sns.despine(bottom=False, top=True, left=True, right=True,
                offset=None, trim=True)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['bottom'].set_color('#555555')
    ax.spines['bottom'].set_position(('axes', 0.0))

    ax.yaxis.grid(True, which="major", linestyle='-',
                  linewidth=0.4, color='#B6B6B6')

    ax.tick_params(axis='x', bottom=True, top=False,
                   width=0.8, labelsize=8, color='#555555')
    ax.tick_params(axis='y', left=False, right=False,
                   width=0.8, labelsize=8, color='#555555')

    hazard_scenario_list = hazards.hazard_scenario_list
    xtick_labels = calc_tick_vals(hazard_scenario_list)

    xtick_pos = []
    for val in xtick_labels:
        xtick_pos.append(hazard_scenario_list.index(val))
    intensity_label = hazards.intensity_measure_param+' ('+\
                      hazards.intensity_measure_unit+')'

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, rotation='vertical')
    ax.set_xlabel(intensity_label, labelpad=9, size=10)

    ax.set_yticks(np.linspace(0.0, 1.0, 11, endpoint=True))
    ax.set_ylabel('Loss Fraction (%)', labelpad=9, size=10)

    ax.set_title('Loss Ratio', loc='center', y=1.04,fontsize=12, weight='bold')

    figfile = os.path.join(scenario.output_path, 'fig_lossratio_boxplot.png')
    plt.margins(0.05)
    plt.savefig(figfile, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)


def write_system_response(response_list, infrastructure, scenario, hazards):
    # ------------------------------------------------------------------------
    # 'ids_comp_vs_haz' is a dict of numpy arrays
    # We pickle it for archival. But the file size can get very large.
    # So we zip it for archival and delete the original
    # ------------------------------------------------------------------------
    idshaz = os.path.join(scenario.raw_output_dir, 'ids_comp_vs_haz.pickle')
    haz_vs_ds_index_of_comp = response_list[0]
    with open(idshaz, 'wb') as handle:
        for response_key in sorted(haz_vs_ds_index_of_comp.keys()):
            pickle.dump(
                {response_key: haz_vs_ds_index_of_comp[response_key]},
                handle, pickle.HIGHEST_PROTOCOL
                )

    idshaz_zip = os.path.join(scenario.raw_output_dir, 'ids_comp_vs_haz.zip')
    zf = zipfile.ZipFile(idshaz_zip, mode='w', allowZip64=True)
    zf.write(idshaz, compress_type=zipfile.ZIP_DEFLATED)
    zf.close()
    os.remove(idshaz)

    # ------------------------------------------------------------------------
    # System output file (for given hazard transfer parameter value)
    # ------------------------------------------------------------------------
    sys_output_dict = response_list[1]
    sod_pkl = os.path.join(scenario.raw_output_dir,
                           'sys_output_dict.pickle')
    with open(sod_pkl, 'wb') as handle:
        for response_key in sorted(sys_output_dict.keys()):
            pickle.dump(
                {response_key: sys_output_dict[response_key]},
                handle, pickle.HIGHEST_PROTOCOL
                )

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
    crd_pkl = os.path.join(scenario.raw_output_dir,'component_resp_dict.pickle')
    with open(crd_pkl, 'wb') as handle:
        for response_key in sorted(component_resp_dict.keys()):
            pickle.dump(
                {response_key: component_resp_dict[response_key]},
                handle, pickle.HIGHEST_PROTOCOL
                )

    # ------------------------------------------------------------------------
    # Hazard response for component types
    # ------------------------------------------------------------------------
    comptype_resp_dict = response_list[3]
    outfile_comptype_resp = os.path.join(
        scenario.output_path, 'comptype_response.csv')
    comptype_resp_df = pd.DataFrame(comptype_resp_dict)
    comptype_resp_df.index.names = ['component_type', 'response']
    comptype_resp_df.to_csv(
        outfile_comptype_resp, sep=',',
        index_label=['component_type', 'response'])

    # ------------------------------------------------------------------------
    # Calculating system fragility:
    # ------------------------------------------------------------------------

    # infrastructure econ loss for sample
    economic_loss_array = response_list[5]
    sys_frag = np.zeros_like(economic_loss_array, dtype=int)
    sys_damage_state_bounds = infrastructure.get_dmg_scale_bounds()
    for j, hazard_level in enumerate(hazards.hazard_scenario_list):
        for i in range(scenario.num_samples):
            # system output and economic loss
            sys_frag[i, j] = \
                np.sum(economic_loss_array[i, j] > sys_damage_state_bounds)

    # Calculating Probability of Exceedence:
    pe_sys_econloss = np.zeros(
        (len(infrastructure.get_system_damage_states()),
         hazards.num_hazard_pts)
        )
    for j in range(hazards.num_hazard_pts):
        for i in range(len(infrastructure.get_system_damage_states())):
            pe_sys_econloss[i, j] = \
                np.sum(sys_frag[:, j] >= i) / float(scenario.num_samples)

    compcls_dmg_level_percentages = response_list[6]
    comp_class_list = infrastructure.get_component_classes()
    pe_sys_classdmg = np.zeros(
        (len(infrastructure.get_system_damage_states()),
         hazards.num_hazard_pts)
        )

    ###########################################################################
    # print("****************************")
    # print('compcls_dmg_level_percentages')
    # pp.pprint(compcls_dmg_level_percentages)

    # for j in range(hazards.num_hazard_pts):
    #     for i in range(len(infrastructure.get_system_damage_states())):
    #         pe_sys_classdmg[i, j] = \
    #
    ###########################################################################

    np.save(os.path.join(scenario.raw_output_dir, 'sys_frag.npy'), sys_frag)
    np.save(os.path.join(scenario.raw_output_dir, 'pe_sys_econloss.npy'),
            pe_sys_econloss)
# ------------------------------------------------------------------------------


def pe_by_component_class(response_list, infrastructure, scenario, hazards):
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
    #   Damage state boundaries for Component Type Failures (Substations) are
    #   based on HAZUS MH MR3, p 8-66 to 8-68
    # ------------------------------------------------------------------------

    cp_classes_in_system = np.unique(list(infrastructure.
                                          get_component_class_list()))

    cp_class_map = {k: [] for k in cp_classes_in_system}
    for comp_id, component in list(infrastructure.components.items()):
        cp_class_map[component.component_class].append(component)

    if infrastructure.system_class == 'Substation':
        cp_classes_costed = \
            [x for x in cp_classes_in_system
             if x not in infrastructure.uncosted_classes]

        # --- System fragility - Based on Failure of Component Classes ---
        comp_class_failures = \
            {cc: np.zeros((scenario.num_samples, hazards.num_hazard_pts))
             for cc in cp_classes_costed}

        comp_class_frag = \
            {cc: np.zeros((scenario.num_samples, hazards.num_hazard_pts))
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

        for j, (scenario_name, hazard_data) in \
                enumerate(hazards.scenario_hazard_data.items()):
            for i in range(scenario.num_samples):
                for compclass in cp_classes_costed:
                    for comptype in cp_class_map[compclass]:
                        comp_ndx = list(infrastructure.components.keys()).\
                            index(comptype.component_id)
                        # -----------------------------------------------------
                        if response_list[0][scenario_name][i, comp_ndx] >= 2:
                            comp_class_failures[compclass][i, j] += 1
                        # comp_class_failures[compclass][i, j] += \
                        #     response_list[0][scenario_name][i, comp_ndx]
                        # -----------------------------------------------------
                    comp_class_failures[compclass][i, j] /= \
                        len(cp_class_map[compclass])

                    comp_class_frag[compclass][i, j] = \
                        np.sum(comp_class_failures[compclass][i, j] > \
                               infrastructure.ds_lims_compclasses[compclass])

        # Probability of Exceedence -- Based on Failure of Component Classes
        pe_sys_cpfailrate = np.zeros(
            (len(infrastructure.sys_dmg_states), hazards.num_hazard_pts)
        )
        for p in range(hazards.num_hazard_pts):
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
                                 hazards.num_hazard_pts))
    for l, hazard in enumerate(hazards.listOfhazards):
        # compute expected damage ratio
        for j, component in enumerate(infrastructure.components.values()):
            # TODO remove invalid Component accesses !!

            component_pe_ds = np.zeros(len(component.damage_states))
            for damage_state_index in component.damage_states.keys():
                x_loc, y_loc = component.get_location()
                hazard_intensity \
                    = hazard.get_hazard_intensity_at_location(x_loc, y_loc)
                component_pe_ds[damage_state_index] \
                    = component.damage_states[damage_state_index].\
                      response_function(hazard_intensity)

            component_pe_ds = component_pe_ds[1:]
            pb = pe2pb(component_pe_ds)
            dr = np.array([component.damage_states[int(ds)].damage_ratio
                           for ds in range(len(component.damage_states))])
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
    economic_loss_array = response_list[5]
    calculated_output_array = response_list[4]

    outdat = {out_cols[0]: hazards.hazard_scenario_list,
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
    component_resp_df.columns = hazards.hazard_scenario_name
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
            os.path.join(scenario.raw_output_dir,
                         'economic_loss_array.npy'),
            economic_loss_array
        )

        np.save(
            os.path.join(scenario.raw_output_dir,
                         'calculated_output_array.npy'),
            calculated_output_array
        )

        np.save(
            os.path.join(scenario.raw_output_dir,
                         'exp_damage_ratio.npy'),
            exp_damage_ratio
        )

    # ... END POST-PROCESSING
    # **************************************************************************

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
