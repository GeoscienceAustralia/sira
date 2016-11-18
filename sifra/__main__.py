#!/usr/bin/env python
__author__ = 'Sudipta Basak'

# from sifraclasses import Scenario, PowerStation, PotableWaterTreatmentPlant
from sifraclasses import *
from sysresponse import calc_sys_output, calc_loss_arrays, post_processing
import sys, os


if __name__ == "__main__":
    SETUPFILE = sys.argv[1]
    discard = {}
    config = {}

    exec (open(SETUPFILE).read(), discard, config)
    SystemObject = eval(config["SYSTEM_CLASS"])
    scn = Scenario(SETUPFILE)
    sysobj = SystemObject(SETUPFILE)

    # Define input files, output location, scenario inputs
    SYS_CONFIG_FILE = os.path.join(scn.input_path,
                                   sysobj.sys_config_file_name)

    if not os.path.exists(scn.output_path):
        os.makedirs(scn.output_path)

    hazard_transfer_label = scn.intensity_measure_param +\
                            ' (' + scn.intensity_measure_unit + ')'

    # cpdict, output_dict, input_dict, \
    #     nodes_by_commoditytype = convert_df_to_dict(sysobj)
    component_resp_df = calc_sys_output(sysobj, scn)
    ids_comp_vs_haz, sys_output_dict, \
        component_resp_dict, calculated_output_array, \
        economic_loss_array, output_array_given_recovery \
        = calc_loss_arrays(sysobj, scn,
                           component_resp_df,
                           parallel_proc=scn.run_parallel_proc)

    post_processing(sysobj, scn,
                    ids_comp_vs_haz,
                    sys_output_dict,
                    component_resp_dict,
                    calculated_output_array,
                    economic_loss_array,
                    output_array_given_recovery)
