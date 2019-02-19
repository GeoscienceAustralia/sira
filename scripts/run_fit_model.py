from __future__ import print_function

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.patheffects as PathEffects
import seaborn as sns

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import lmfit
import pandas as pd
pd.options.display.float_format = '{:,.4f}'.format
import json

import os
import warnings

import sifra.sifraplot as spl

import brewer2mpl
from colorama import Fore, Back, init, Style
init()

import argparse
from sifra.configuration import Configuration
from sifra.scenario import Scenario
from sifra.modelling.hazard import HazardsContainer
from sifra.model_ingest import ingest_model
from sifra.fit_model import fit_prob_exceed_model
# ============================================================================

def main():
    # ------------------------------------------------------------------------
    # Read in SETUP data
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--setup", type=str,
                        help="Setup file for simulation scenario, and \n"
                             "locations of inputs, outputs, and system model.")
    parser.add_argument("-v", "--verbose",  type=str,
                        help="Choose option for logging level from: \n"
                             "DEBUG, INFO, WARNING, ERROR, CRITICAL.")
    parser.add_argument("-d", "--dirfile", type=str,
                        help="JSON file with location of input/output files "
                             "from past simulation that is to be analysed\n.")
    args = parser.parse_args()

    if args.setup is None:
        raise ValueError("Must provide a correct setup argument: "
                         "`-s` or `--setup`,\n"
                         "Setup file for simulation scenario, and \n"
                         "locations of inputs, outputs, and system model.\n")

    if not os.path.exists(args.dirfile):
        raise ValueError("Could not locate file with directory locations"
                         "with results from pre-run simulations.\n")

    # Define input files, output location, scenario inputs
    with open(args.dirfile, 'r') as dat:
        dir_dict = json.load(dat)

    # Configure simulation model.
    # Read data and control parameters and construct objects.
    config = Configuration(args.setup,
                           run_mode='analysis',
                           output_path=dir_dict["OUTPUT_PATH"])
    scenario = Scenario(config)
    hazards = HazardsContainer(config)
    infrastructure = ingest_model(config)

    if not config.SYS_CONF_FILE_NAME == dir_dict["SYS_CONF_FILE_NAME"]:
        raise NameError("Names for supplied system model names did not match."
                        "Aborting.\n")

    # --------------------------------------------------------------------------

    OUTPUT_PATH = dir_dict["OUTPUT_PATH"]
    RAW_OUTPUT_DIR = dir_dict["RAW_OUTPUT_DIR"]
    hazard_scenarios = hazards.hazard_scenario_list

    sys_limit_states = infrastructure.get_system_damage_states()
    # one_comp = infrastructure.components.values()[0]
    # sys_limit_states = [one_comp.damage_states[ds].damage_state_name
    #                     for ds in one_comp.damage_states]

    # Test switches
    FIT_PE_DATA = scenario.fit_pe_data
    # SWITCH_FIT_RESTORATION_DATA = scenario.SWITCH_FIT_RESTORATION_DATA
    # RESTORATION_TIME_RANGE = scenario.restoration_time_range

    # ------------------------------------------------------------------------
    # READ in raw output files from prior analysis of system fragility

    economic_loss_array = \
        np.load(os.path.join(
            RAW_OUTPUT_DIR, 'economic_loss_array.npy'))

    calculated_output_array = \
        np.load(os.path.join(
            RAW_OUTPUT_DIR, 'calculated_output_array.npy'))

    exp_damage_ratio = \
        np.load(os.path.join(
            RAW_OUTPUT_DIR, 'exp_damage_ratio.npy'))

    sys_frag = \
        np.load(os.path.join(
            RAW_OUTPUT_DIR, 'sys_frag.npy'))

    # output_array_given_recovery = \
    #     np.load(os.path.join(
    #         RAW_OUTPUT_DIR, 'output_array_given_recovery.npy'))

    # required_time = \
    #     np.load(os.path.join(RAW_OUTPUT_DIR, 'required_time.npy'))

    # --------------------------------------------------------------------------

    if infrastructure.system_class.lower() == 'powerstation':
        pe_sys = np.load(os.path.join(RAW_OUTPUT_DIR, 'pe_sys_econloss.npy'))
    elif infrastructure.system_class.lower() == 'substation':
        pe_sys = np.load(os.path.join(RAW_OUTPUT_DIR, 'pe_sys_cpfailrate.npy'))
    # elif infrastructure.system_class.lower() == 'substation':
    #     pe_sys = np.load(os.path.join(RAW_OUTPUT_DIR, 'pe_sys_econloss.npy'))
    elif infrastructure.system_class.lower() in [
        "potablewatertreatmentplant", "pwtp",
        "wastewatertreatmentplant", "wwtp",
        "watertreatmentplant", "wtp"]:
        pe_sys = np.load(os.path.join(RAW_OUTPUT_DIR, 'pe_sys_econloss.npy'))

    # --------------------------------------------------------------------------
    # Calculate & Plot Fitted Models

    if FIT_PE_DATA:
        fit_prob_exceed_model(hazard_scenarios,
                              pe_sys,
                              sys_limit_states,
                              OUTPUT_PATH,
                              config)

    # sys_fn = approximate_generic_sys_restoration(sc, fc, sys_frag,
    #                                              output_array_given_recovery)
    #
    # if SWITCH_FIT_RESTORATION_DATA:
    #     sys_rst_mdl_mode1 = fit_restoration_data(
    #         RESTORATION_TIME_RANGE, sys_fn, sys_limit_states, sc.output_path)
    #     # sys_rst_mdl_mode2 = fit_restoration_data_multimode(
    #     #     RESTORATION_TIME_RANGE, sys_fn, sys_limit_states, scn.output_path)
    #     print("\n" + "-" * 79)

# ============================================================================

if __name__ == "__main__":
    if __name__ == "__main__":
        print()
        print(Fore.CYAN + Back.BLACK + Style.BRIGHT +
              ">> Initiating attempt to fit model to simulation data ... " +
              Style.RESET_ALL + "\n")
        main()
