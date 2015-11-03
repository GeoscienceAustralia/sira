#!/usr/bin/env python
__author__ = 'Sudipta Basak'

import sys, os
from siraclasses import Scenario, Facility
from sira import power_calc, calc_loss_arrays, post_porcessing


if __name__ == "__main__":
    SETUPFILE = sys.argv[1]
    discard = {}
    config = {}
    execfile(SETUPFILE, discard, config)
    sc = Scenario(SETUPFILE)
    fc = Facility(SETUPFILE)

    # Define input files, output location, scenario inputs
    INPUT_PATH = os.path.join(os.getcwd(), sc.input_dir_name)
    SYS_CONFIG_FILE = os.path.join(INPUT_PATH, fc.sys_config_file_name)

    if not os.path.exists(sc.output_dir_name):
        os.makedirs(sc.output_dir_name)

    hazard_transfer_label = sc.hazard_transfer_param + ' (' + sc.hazard_transfer_unit+ ')'

    # cpdict, output_dict, input_dict, nodes_by_commoditytype = convert_df_to_dict(fc)
    component_resp_df = power_calc(fc, sc)
    ids_comp_vs_haz, sys_output_dict, component_resp_dict, calculated_output_array, \
        economic_loss_array, output_array_given_recovery \
            = calc_loss_arrays(fc, sc, component_resp_df, parallel_or_serial=sc.parallel_or_serial)

    post_porcessing(fc, sc, ids_comp_vs_haz, sys_output_dict, component_resp_dict, calculated_output_array,
                    economic_loss_array, output_array_given_recovery)


