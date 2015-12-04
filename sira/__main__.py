#!/usr/bin/env python
__author__ = 'Sudipta Basak'

import sys, os
from siraclasses import Scenario, Facility, PowerStation
from sira import power_calc, calc_loss_arrays, post_processing


if __name__ == "__main__":
    SETUPFILE = sys.argv[1]
    discard = {}
    config = {}

    exec (open(SETUPFILE).read(), discard, config)
    FacilityObj = eval(config["SYSTEM_CLASS"])
    sc = Scenario(SETUPFILE)
    fc = FacilityObj(SETUPFILE)

    # Define input files, output location, scenario inputs
    INPUT_PATH = os.path.join(os.getcwd(), sc.input_dir_name)
    SYS_CONFIG_FILE = os.path.join(INPUT_PATH, fc.sys_config_file_name)

    if not os.path.exists(sc.output_path):
        os.makedirs(sc.output_path)

    hazard_transfer_label = sc.hazard_transfer_param +\
                            ' (' + sc.hazard_transfer_unit+ ')'

    # cpdict, output_dict, input_dict, \
    #     nodes_by_commoditytype = convert_df_to_dict(fc)
    component_resp_df = power_calc(fc, sc)
    ids_comp_vs_haz, sys_output_dict, \
        component_resp_dict, calculated_output_array, \
        economic_loss_array, output_array_given_recovery \
        = calc_loss_arrays(fc, sc,
                           component_resp_df,
                           parallel_or_serial=sc.parallel_or_serial)

    post_processing(fc, sc,
                    ids_comp_vs_haz,
                    sys_output_dict,
                    component_resp_dict,
                    calculated_output_array,
                    economic_loss_array,
                    output_array_given_recovery)


