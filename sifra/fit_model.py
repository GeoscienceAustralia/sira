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

# ----------------------------------------------------------------------------
# For plots: using the  brewer2 color maps by Cynthia Brewer
# ----------------------------------------------------------------------------

set2 = brewer2mpl.get_map('Set2', 'qualitative', 5).mpl_colors
markers = ['o', '^', 's', 'D', 'x', '+']

def rectify_arry(number):

    newarry = []
    for num in number:
        newarry.append(rectify_limits(num))

    return newarry

def rectify_limits(number):

    if type(number) is list:
        return rectify_arry(number)
    if type(number) is np.ndarray:
        return rectify_arry(number)



    if number < 0:
        return 0

    if number < 1:
        return 1

    if number > 10000:
        return 10000

    return number
# ============================================================================
#
# PROBABILITY of EXCEEDANCE MODEL FITTING
#
# ----------------------------------------------------------------------------
# LOGNORMAL CURVE FITTING
#
# Parameters in scipy LOGNORMAL distribution:
#
# shape = sigma = log(s)      s = geometric standard deviation
#                             sigma = standard deviation of log(X)
#
# scale = M = exp(mu)         M = geometric mean == median
#                             mu = mean of log(X) = log(scale)
#
# location (keyword 'loc') shifts the distribution to the left or right.
# Unless data contain negative values, the location parameter is fixed at 0.
# During curve fitting, this can be set by using floc=0.
#
# Note on the covariance matrix returned by scipy.optimize.curve_fit:
# The square root of the diagonal values are the 1-sigma uncertainties of
# the fit parameters.
# ----------------------------------------------------------------------------

def lognormal_cdf(x, median, logstd, loc=0):
    scale = median
    shape = logstd
    loc = loc
    return stats.lognorm.cdf(x, shape, loc=loc, scale=scale)


def res_lognorm_cdf(params, x, data, eps=None):
    shape = params['logstd'].value
    scale = params['median'].value
    loc = params['loc'].value
    model = stats.lognorm.cdf(x, shape, loc=loc, scale=scale)
    if eps is None:
        return (model - data)
    return (model - data) / eps


def fit_prob_exceed_model(hazard_input_vals, pb_exceed, SYS_DS,
                          out_path, config):
    """
    Fit a Lognormal CDF model to simulated probability exceedance data

    :param hazard_input_vals: input values for hazard intensity (numpy array)
    :param pb_exceed: probability of exceedance (2D numpy array)
    :param SYS_DS: discrete damage states (list)
    :param out_path: directory path for writing output (string)
    :returns:  fitted exceedance model parameters (PANDAS dataframe)
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # DataFrame for storing the calculated System Damage Algorithms for
    # exceedence probabilities.

    indx = pd.Index(SYS_DS[1:], name='Damage States')
    sys_dmg_model = pd.DataFrame(index=indx, columns=['Median', 'LogStdDev', 'Location', 'Chi-Sqr'])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # INITIAL FIT

    sys_dmg_fitted_params = [[] for _ in range(len(SYS_DS))]
    hazard_input_vals = [float(x) for x in hazard_input_vals]
    params_pe = []

    for dx in range(0, len(SYS_DS)):

        x_sample = rectify_limits(hazard_input_vals)
        y_sample = rectify_limits(pb_exceed[dx])

        p0m = rectify_limits(np.mean(y_sample))
        p0s = rectify_limits(np.std(y_sample))

        # Fit the dist:
        params_pe.append(lmfit.Parameters())
        params_pe[dx].add('median', value=p0m,min=0, max=10)  # , min=0, max=10)
        params_pe[dx].add('logstd', value=p0s,min=0, max=10)
        params_pe[dx].add('loc', value=0.0, vary=False,min=0, max=10)


        if dx >= 1:

            print("params_pe[dx]",params_pe[dx])

            sys_dmg_fitted_params[dx] = lmfit.minimize(res_lognorm_cdf, params_pe[dx], args=(x_sample, y_sample))
            sys_dmg_model.loc[SYS_DS[dx]] = (sys_dmg_fitted_params[dx].params['median'].value, sys_dmg_fitted_params[dx].params['logstd'].value, sys_dmg_fitted_params[dx].params['loc'].value, sys_dmg_fitted_params[dx].chisqr)


    print("\n" + "-" * 79)
    print(Fore.YELLOW + "Fitting system FRAGILITY data: Lognormal CDF" + Fore.RESET)
    print("-" * 79)
    # sys_dmg_model = sys_dmg_model.round(decimals)
    print("INITIAL System Fragilities:\n\n", sys_dmg_model, '\n')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Check for crossover and correct as needed

    CROSSOVER_THRESHOLD = 0.001
    CROSSOVER_CORRECTION = True
    if CROSSOVER_CORRECTION:
        sys_dmg_fitted_params = correct_crossover(SYS_DS, pb_exceed, hazard_input_vals, sys_dmg_fitted_params, CROSSOVER_THRESHOLD)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Finalise damage function parameters

    for dx in range(1, len(SYS_DS)):
        sys_dmg_model.loc[SYS_DS[dx]] = sys_dmg_fitted_params[dx].params['median'].value, sys_dmg_fitted_params[dx].params['logstd'].value, sys_dmg_fitted_params[dx].params['loc'].value, sys_dmg_fitted_params[dx].chisqr

    print("\nFINAL System Fragilities: \n")
    print(sys_dmg_model)
    print()



    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Write fitted model params to file
    sys_dmg_model.to_csv(os.path.join(out_path, 'system_model_fragility.csv'), sep=',')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot the simulation data

    plot_data_model(SYS_DS, hazard_input_vals, sys_dmg_model, pb_exceed, out_path, config, PLOT_DATA=True, PLOT_MODEL=False, PLOT_EVENTS=False)
    plot_data_model(SYS_DS, hazard_input_vals, sys_dmg_model, pb_exceed, out_path, config, PLOT_DATA=True, PLOT_MODEL=True, PLOT_EVENTS=False)
    plot_data_model(SYS_DS, hazard_input_vals, sys_dmg_model, pb_exceed, out_path, config, PLOT_DATA=False, PLOT_MODEL=True, PLOT_EVENTS=True)

    # RETURN a DataFrame with the fitted model parameters
    return sys_dmg_model

# ==============================================================================

def plot_data_model(SYS_DS, hazard_input_vals, sys_dmg_model, pb_exceed, out_path, config, PLOT_DATA=True, PLOT_MODEL=True, PLOT_EVENTS=False):

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if sum([PLOT_DATA, PLOT_MODEL, PLOT_EVENTS])==0:
        raise AttributeError

    sns.set(style="darkgrid")

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    colours = spl.ColourPalettes()
    COLR_DS = colours.FiveLevels[-1*len(SYS_DS):]


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # [Plot 1 of 3] The Data Points

    if PLOT_DATA:
        outfig = os.path.join(out_path, 'fig_sys_pe_DATA.png')
        spl.add_legend_subtitle("$\\bf{DATA}$")
        for i in range(1, len(SYS_DS)):
            ax.plot(hazard_input_vals, pb_exceed[i], label=SYS_DS[i], clip_on=False, color=COLR_DS[i], linestyle='solid', alpha=0.4, marker=markers[i - 1], markersize=3)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # [Plot 2 of 3] The Fitted Model

    if PLOT_MODEL:
        outfig = os.path.join(out_path, 'fig_sys_pe_MODEL.png')
        spl.add_legend_subtitle("\n$\\bf{FITTED\ MODEL}$")
        xmax = max(hazard_input_vals)
        xformodel = np.linspace(0, xmax, 101, endpoint=True)
        dmg_mdl_arr = np.zeros((len(SYS_DS), len(xformodel)))

        for dx in range(1, len(SYS_DS)):
            shape = sys_dmg_model.loc[SYS_DS[dx], 'LogStdDev']
            loc = sys_dmg_model.loc[SYS_DS[dx], 'Location']
            scale = sys_dmg_model.loc[SYS_DS[dx], 'Median']
            dmg_mdl_arr[dx] = stats.lognorm.cdf(xformodel, shape, loc=loc, scale=scale)
            ax.plot(xformodel, dmg_mdl_arr[dx], label=SYS_DS[dx], clip_on=False, color=COLR_DS[dx], alpha=0.65, linestyle='solid', linewidth=1.6)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # [Plot 3 of 3] The Scenario Events

    if PLOT_EVENTS:
        outfig = os.path.join(out_path, 'fig_sys_pe_MODEL_with_scenarios.png')
        spl.add_legend_subtitle("\n$\\bf{EVENTS}$")
        for i, haz in enumerate(config.FOCAL_HAZARD_SCENARIOS):
            event_num = str(i+1)
            event_intensity_str = "{:.3f}".format(float(haz))
            event_color = colours.BrewerSpectral[i]
            try:
                event_label = event_num + ". " + config.FOCAL_HAZARD_SCENARIO_NAMES[i]+" : " + event_intensity_str
            except:
                event_label = event_num + " : " + event_intensity_str

            ax.plot(float(haz), 0,label=event_label,color=event_color, marker='', markersize=2, linestyle=('solid'))
            ax.plot(float(haz), 1.04, label='', clip_on=False, color=event_color, marker='o', fillstyle='none', markersize=12, linestyle='solid', markeredgewidth=1.0)
            ax.annotate(event_num,  xy=(float(haz), 0), xycoords='data', xytext=(float(haz), 1.038), textcoords='data', ha='center', va='center', rotation=0, size=8, fontweight='bold', color=event_color, annotation_clip=False, bbox=dict(boxstyle='round, pad=0.2', fc='yellow', alpha=0.0), path_effects=[PathEffects.withStroke(linewidth=2, foreground="w")],arrowprops=dict(arrowstyle='-|>, head_length=0.5, head_width=0.3', shrinkA=3.0, shrinkB=0.0, connectionstyle='arc3,rad=0.0', color=event_color, alpha=0.8, linewidth=1.0, linestyle=('solid'),path_effects=[PathEffects.withStroke(linewidth=2.5, foreground="w")]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    figtitle = 'System Fragility: ' + config.SYSTEM_CLASS

    x_lab = config.INTENSITY_MEASURE_PARAM + ' (' + config.INTENSITY_MEASURE_UNIT + ')'
    y_lab = 'P($D_s$ > $d_s$)'

    y_tick_pos = np.linspace(0.0, 1.0, num=6, endpoint=True)
    y_tick_val = ['{:.1f}'.format(i) for i in y_tick_pos]
    x_tick_pos = np.linspace(0.0, max(hazard_input_vals), num=6, endpoint=True)
    x_tick_val = ['{:.2f}'.format(i) for i in x_tick_pos]

    ax.set_title(figtitle, loc='center', y=1.06, fontweight='bold', size=11)
    ax.set_xlabel(x_lab, size=8, labelpad=10)
    ax.set_ylabel(y_lab, size=8, labelpad=10)

    ax.set_xlim(0, max(x_tick_pos))
    ax.set_xticks(x_tick_pos)
    ax.set_xticklabels(x_tick_val, size=7)
    ax.set_ylim(0, max(y_tick_pos))
    ax.set_yticks(y_tick_pos)
    ax.set_yticklabels(y_tick_val, size=7)
    ax.margins(0, 0)

    # Shrink current axis width by 15%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    # Put a legend to the right of the current axis
    ax.legend(title='', loc='upper left', ncol=1, bbox_to_anchor=(1.02, 1.0), frameon=0, prop={'size': 7})

    plt.savefig(outfig, format='png', dpi=300)
    plt.close(fig)

# ==============================================================================

def correct_crossover(SYS_DS, pb_exceed, x_sample, sys_dmg_fitted_params, CROSSOVER_THRESHOLD=0.001):
    print(Fore.GREEN + "Checking for crossover ..." + Fore.RESET)
    params_pe = lmfit.Parameters()
    for dx in range(1, len(SYS_DS)):
        x_sample = x_sample
        y_sample = pb_exceed[dx]

        mu_hi = sys_dmg_fitted_params[dx].params['median'].value
        sd_hi = sys_dmg_fitted_params[dx].params['logstd'].value
        loc_hi = sys_dmg_fitted_params[dx].params['loc'].value

        y_model_hi = stats.lognorm.cdf(x_sample, sd_hi, loc=loc_hi, scale=mu_hi)

        params_pe.add('median', value=mu_hi, min=0, max=10)
        params_pe.add('logstd', value=sd_hi, min=0, max=10)
        params_pe.add('loc', value=0.0, vary=False, min=0, max=10)
        sys_dmg_fitted_params[dx] = lmfit.minimize(res_lognorm_cdf, params_pe, args=(x_sample, y_sample))

        ####################################################################
        if dx >= 2:
            mu_lo = sys_dmg_fitted_params[dx-1].params['median'].value
            sd_lo = sys_dmg_fitted_params[dx-1].params['logstd'].value
            loc_lo = sys_dmg_fitted_params[dx-1].params['loc'].value
            y_model_lo = stats.lognorm.cdf(x_sample, sd_lo, loc=loc_lo, scale=mu_lo)

            if abs(min(y_model_lo - y_model_hi)) > CROSSOVER_THRESHOLD:
                print(Fore.MAGENTA + "There is overlap for curve pair : " + SYS_DS[dx - 1] + '-' + SYS_DS[dx] + Fore.RESET)

                # Test if higher curve is co-incident with, or precedes lower curve
                if (mu_hi <= mu_lo) or (loc_hi < loc_lo):
                    print(" *** Mean of higher curve too low: resampling")
                    params_pe.add('median', value=mu_hi, min=mu_lo)
                    sys_dmg_fitted_params[dx] = lmfit.minimize( res_lognorm_cdf, params_pe, args=(x_sample, y_sample))

                    (mu_hi, sd_hi, loc_hi) = (sys_dmg_fitted_params[dx].params['median'].value, sys_dmg_fitted_params[dx].params['logstd'].value, sys_dmg_fitted_params[dx].params['loc'].value)

                # Thresholds for testing top or bottom crossover
                delta_top = (3.0 * sd_lo - (mu_hi - mu_lo)) / 3
                delta_btm = (3.0 * sd_lo + (mu_hi - mu_lo)) / 3

                # Test for top crossover: resample if crossover detected
                if (sd_hi < sd_lo) and (sd_hi <= delta_top):
                    print(" *** Attempting to correct upper crossover")
                    params_pe.add('logstd', value=sd_hi, min=delta_top)
                    sys_dmg_fitted_params[dx] = lmfit.minimize(res_lognorm_cdf, params_pe, args=(x_sample, y_sample))

                # Test for bottom crossover: resample if crossover detected
                elif sd_hi >= delta_btm:
                    print(" *** Attempting to correct lower crossover")
                    params_pe.add('logstd', value=sd_hi, max=delta_btm)
                    sys_dmg_fitted_params[dx] = lmfit.minimize(res_lognorm_cdf, params_pe, args=(x_sample, y_sample))

            else:
                print(Fore.GREEN + "There is NO overlap for given THRESHOLD of " + str(CROSSOVER_THRESHOLD) + ", for curve pair: " + SYS_DS[dx - 1] + '-' + SYS_DS[dx] + Fore.RESET)

    return sys_dmg_fitted_params

# ============================================================================
#
# NORMAL CURVE FITTING
#
# ----------------------------------------------------------------------------
# Parameters in scipy NORMAL distribution:
#
# The location (loc) keyword specifies the mean.
# The scale (scale) keyword specifies the standard deviation.
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
#
# Note on the covariance matrix returned by scipy.optimize.curve_fit:
# The square root of the diagonal values are the 1-sigma uncertainties of
# the fit parameters
# ----------------------------------------------------------------------------

def norm_cdf(x, mu, sd):
    return stats.norm.cdf(x, loc=mu, scale=sd)
