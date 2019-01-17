from __future__ import print_function

import matplotlib.pyplot as plt
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

# stream = AnsiToWin32(sys.stderr).stream

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

# def ci_dict_to_df(ci):
#     convp = lambda x: ('%.2f' % (x[0] * 100.0)) + '%'
#     conv = lambda x: x[1]
#     ci_header = []
#     ci_values = []
#     title_set = False
#     for name, row in ci.items():
#         if not title_set:
#             ciheader = [i for i in map(convp, row)]
#             title_set = True
#         ci_values.append([i for i in map(conv, row)])
#     ci_df = pd.DataFrame(ci_values, index=ci.keys(), columns=ci_header)
#     ci_df = ci_df.sort()
#     return ci_df


# ----------------------------------------------------------------------------
# For plots: using the  brewer2 color maps by Cynthia Brewer
# ----------------------------------------------------------------------------

# clrs = brewer2mpl.get_map('RdYlGn', 'Diverging', 11).mpl_colors
set2 = brewer2mpl.get_map('Set2', 'qualitative', 5).mpl_colors

markers = ['o', '^', 's', 'D', 'x', '+']


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

# ==============================================================================

def fit_prob_exceed_model_V2(
        hazard_input_vals, pb_exceed, SYS_DS, out_path, config):
    """
    Fit a Lognormal CDF model to simulated probability exceedance data

    :param hazard_input_vals: input values for hazard intensity (numpy array)
    :param pb_exceed: probability of exceedance (2D numpy array)
    :param SYS_DS: discrete damage states (list)
    :param out_path: directory path for writing output (string)
    :param config: object holding simulation configuration parameters
    :returns:  fitted exceedance model parameters (PANDAS dataframe)
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # DataFrame for storing the calculated System Damage Algorithms for
    # exceedence probabilities.

    indx = pd.Index(SYS_DS[1:], name='Damage States')
    sys_dmg_model = pd.DataFrame(index=indx,
                                 columns=['Median',
                                          'LogStdDev',
                                          'Location',
                                          'Chi-Sqr'])

    pex_model = lmfit.Model(lognormal_cdf)
    print(pex_model.param_names)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # INITIAL FIT

    sys_dmg_fitted_params = [[] for _ in range(len(SYS_DS))]
    hazard_input_vals = [float(x) for x in hazard_input_vals]
    params_pe = []

    for dx in range(0, len(SYS_DS)):
    # for dx, dsname in enumerate(SYS_DS[1:]):
        x_sample = hazard_input_vals
        y_sample = pb_exceed[dx]

        p0m = np.mean(y_sample)
        p0s = np.std(y_sample)

        # Fit the dist:
        params_pe.append(lmfit.Parameters())
        params_pe[dx].add('median', value=p0m)  # , min=0, max=10)
        params_pe[dx].add('logstd', value=p0s)
        params_pe[dx].add('loc', value=0.0, vary=False)

        if dx >= 1:

            sys_dmg_fitted_params[dx] = lmfit.minimize(
                res_lognorm_cdf, params_pe[dx], args=(x_sample, y_sample))

            sys_dmg_model.loc[SYS_DS[dx]] \
                = (sys_dmg_fitted_params[dx].params['median'].value,
                   sys_dmg_fitted_params[dx].params['logstd'].value,
                   sys_dmg_fitted_params[dx].params['loc'].value,
                   sys_dmg_fitted_params[dx].chisqr)

    print("\n" + "-" * 79)
    print(Fore.YELLOW +
          "Fitting system FRAGILITY data: Lognormal CDF" +
          Fore.RESET)
    print("-" * 79)
    # sys_dmg_model = sys_dmg_model.round(decimals)
    print("INITIAL System Fragilities:\n\n",
          sys_dmg_model, '\n')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Check for crossover and correct as needed

    CROSSOVER_THRESHOLD = 0.001
    CROSSOVER_CORRECTION = True
    if CROSSOVER_CORRECTION:
        sys_dmg_fitted_params = correct_crossover(
            SYS_DS, pb_exceed, hazard_input_vals, sys_dmg_fitted_params,
            CROSSOVER_THRESHOLD)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Finalise damage function parameters

    for dx in range(1, len(SYS_DS)):
        sys_dmg_model.loc[SYS_DS[dx]] = \
            sys_dmg_fitted_params[dx].params['median'].value, \
            sys_dmg_fitted_params[dx].params['logstd'].value, \
            sys_dmg_fitted_params[dx].params['loc'].value, \
            sys_dmg_fitted_params[dx].chisqr

    print("\nFINAL System Fragilities: \n")
    print(sys_dmg_model)
    print()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Write fitted model params to file

    sys_dmg_model.to_csv(
        os.path.join(out_path, 'system_model_fragility.csv'), sep=',')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot the simulation data

    plot_data_model(SYS_DS,
                    hazard_input_vals,
                    sys_dmg_model,
                    pb_exceed,
                    out_path,
                    config,
                    PLOT_DATA=True,
                    PLOT_MODEL=False,
                    PLOT_EVENTS=False)
    plot_data_model(SYS_DS,
                    hazard_input_vals,
                    sys_dmg_model,
                    pb_exceed,
                    out_path,
                    config,
                    PLOT_DATA=True,
                    PLOT_MODEL=True,
                    PLOT_EVENTS=False)
    plot_data_model(SYS_DS,
                    hazard_input_vals,
                    sys_dmg_model,
                    pb_exceed,
                    out_path,
                    config,
                    PLOT_DATA=True,
                    PLOT_MODEL=True,
                    PLOT_EVENTS=True)

    # RETURN a DataFrame with the fitted model parameters
    return sys_dmg_model

# ==============================================================================

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
    sys_dmg_model = pd.DataFrame(index=indx,
                                 columns=['Median',
                                          'LogStdDev',
                                          'Location',
                                          'Chi-Sqr'])
    # decimals = pd.Series([3, 3, 3],
    #                      index=['Median', 'LogStdDev', 'Location'])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # INITIAL FIT

    # sys_dmg_ci = [{} for _ in xrange(len(SYS_DS))]
    sys_dmg_fitted_params = [[] for _ in range(len(SYS_DS))]
    hazard_input_vals = [float(x) for x in hazard_input_vals]
    params_pe = []

    for dx in range(0, len(SYS_DS)):
    # for dx, dsname in enumerate(SYS_DS[1:]):
        x_sample = hazard_input_vals
        y_sample = pb_exceed[dx]

        p0m = np.mean(y_sample)
        p0s = np.std(y_sample)

        # Fit the dist:
        params_pe.append(lmfit.Parameters())
        params_pe[dx].add('median', value=p0m)  # , min=0, max=10)
        params_pe[dx].add('logstd', value=p0s)
        params_pe[dx].add('loc', value=0.0, vary=False)

        if dx >= 1:

            sys_dmg_fitted_params[dx] = lmfit.minimize(res_lognorm_cdf, params_pe[dx], args=(x_sample, y_sample))

            sys_dmg_model.loc[SYS_DS[dx]] \
                = (sys_dmg_fitted_params[dx].params['median'].value,
                   sys_dmg_fitted_params[dx].params['logstd'].value,
                   sys_dmg_fitted_params[dx].params['loc'].value,
                   sys_dmg_fitted_params[dx].chisqr)

    # sys_dmg_model['Median'] = sys_dmg_model['Median'].map('{:,.3f}'.format)
    # sys_dmg_model['LogStdDev'] = sys_dmg_model['LogStdDev'].map('{:,.3f}'.format)
    # sys_dmg_model['Location'] = sys_dmg_model['Location'].map('{:,.1f}'.format)

    print("\n" + "-" * 79)
    print(Fore.YELLOW +
          "Fitting system FRAGILITY data: Lognormal CDF" +
          Fore.RESET)
    print("-" * 79)
    # sys_dmg_model = sys_dmg_model.round(decimals)
    print("INITIAL System Fragilities:\n\n",
          sys_dmg_model, '\n')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Check for crossover and correct as needed

    CROSSOVER_THRESHOLD = 0.001
    CROSSOVER_CORRECTION = True
    if CROSSOVER_CORRECTION:
        sys_dmg_fitted_params = correct_crossover(
            SYS_DS, pb_exceed, hazard_input_vals, sys_dmg_fitted_params,
            CROSSOVER_THRESHOLD)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Finalise damage function parameters

    for dx in range(1, len(SYS_DS)):
        sys_dmg_model.loc[SYS_DS[dx]] = \
            sys_dmg_fitted_params[dx].params['median'].value, \
            sys_dmg_fitted_params[dx].params['logstd'].value, \
            sys_dmg_fitted_params[dx].params['loc'].value, \
            sys_dmg_fitted_params[dx].chisqr
            # sys_dmg_ci[dx] = lmfit.conf_interval(
            #     sys_dmg_fitted_params[dx], sigmas=[0.674,0.950,0.997])

    print("\nFINAL System Fragilities: \n")
    print(sys_dmg_model)
    print()

    # for dx in range(1, len(SYS_DS)):
    #     print("\n\nFragility model statistics for damage state: %s"
    #           % SYS_DS[dx])
    #     print("Goodness-of-Fit chi-square test statistic: %f"
    #           % sys_dmg_fitted_params[dx].chisqr)
    #     print("Confidence intervals: ")
    #     lmfit.printfuncs.report_ci(sys_dmg_ci[dx])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Write fitted model params to file

    sys_dmg_model.to_csv(
        os.path.join(out_path, 'system_model_fragility.csv'), sep=',')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot the simulation data

    plot_data_model(SYS_DS,
                    hazard_input_vals,
                    sys_dmg_model,
                    pb_exceed,
                    out_path,
                    config,
                    PLOT_DATA=True,
                    PLOT_MODEL=False,
                    PLOT_EVENTS=False)
    plot_data_model(SYS_DS,
                    hazard_input_vals,
                    sys_dmg_model,
                    pb_exceed,
                    out_path,
                    config,
                    PLOT_DATA=True,
                    PLOT_MODEL=True,
                    PLOT_EVENTS=False)
    plot_data_model(SYS_DS,
                    hazard_input_vals,
                    sys_dmg_model,
                    pb_exceed,
                    out_path,
                    config,
                    PLOT_DATA=False,
                    PLOT_MODEL=True,
                    PLOT_EVENTS=True)

    # RETURN a DataFrame with the fitted model parameters
    return sys_dmg_model

# ==============================================================================

def plot_data_model(SYS_DS,
                    hazard_input_vals,
                    sys_dmg_model,
                    pb_exceed,
                    out_path,
                    config,
                    PLOT_DATA=True,
                    PLOT_MODEL=True,
                    PLOT_EVENTS=False
                    ):

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if sum([PLOT_DATA, PLOT_MODEL, PLOT_EVENTS])==0:
        raise AttributeError

    sns.set(style="darkgrid")

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    colours = spl.ColourPalettes()
    COLR_DS = colours.FiveLevels[-1*len(SYS_DS):]
    # grid_colr = '#B6B6B6'
    # spine_colr = 'black'

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # [Plot 1 of 3] The Data Points

    if PLOT_DATA:
        outfig = os.path.join(out_path, 'fig_sys_pe_DATA.png')
        spl.add_legend_subtitle("$\\bf{DATA}$")
        for i in range(1, len(SYS_DS)):
            ax.plot(hazard_input_vals,
                    pb_exceed[i],
                    label=SYS_DS[i],
                    clip_on=False,
                    color=COLR_DS[i],
                    linestyle='',
                    alpha=0.4,
                    marker=markers[i - 1],
                    markersize=3)

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
            dmg_mdl_arr[dx] = stats.lognorm.cdf(
                xformodel, shape, loc=loc, scale=scale)
            ax.plot(xformodel,
                    dmg_mdl_arr[dx],
                    label=SYS_DS[dx], clip_on=False,
                    color=COLR_DS[dx], alpha=0.65,
                    linestyle='-', linewidth=1.6)

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
                event_label = event_num + ". " + \
                              config.FOCAL_HAZARD_SCENARIO_NAMES[i]+\
                              " : " + event_intensity_str
            except:
                event_label = event_num + " : " + event_intensity_str

            ax.plot(float(haz), 0,
                    label=event_label,
                    color=event_color,
                    marker='',
                    markersize=2,
                    linestyle='-')
            ax.plot(float(haz), 1.04,
                    label='',
                    clip_on=False,
                    color=event_color,
                    marker='o',
                    fillstyle='none',
                    markersize=12,
                    linestyle='-',
                    markeredgewidth=1.0)
            ax.annotate(
                event_num, #event_intensity_str,
                xy=(float(haz), 0), xycoords='data',
                xytext=(float(haz), 1.038), textcoords='data',
                ha='center', va='center', rotation=0,
                size=8, fontweight='bold', color=event_color,
                annotation_clip=False,
                bbox=dict(boxstyle='round, pad=0.2', fc='yellow', alpha=0.0),
                path_effects=\
                    [PathEffects.withStroke(linewidth=2, foreground="w")],
                arrowprops=dict(
                    arrowstyle='-|>, head_length=0.5, head_width=0.3',
                    shrinkA=3.0,
                    shrinkB=0.0,
                    connectionstyle='arc3,rad=0.0',
                    color=event_color,
                    alpha=0.8,
                    linewidth=1.0,
                    linestyle="-",
                    path_effects=\
                        [PathEffects.withStroke(linewidth=2.5, foreground="w")]
                    )
                )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    figtitle = 'System Fragility: ' + config.SYSTEM_CLASS

    x_lab = config.INTENSITY_MEASURE_PARAM + ' (' + \
            config.INTENSITY_MEASURE_UNIT + ')'
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
    ax.set_position([box.x0,
                     box.y0,
                     box.width * 0.85,
                     box.height])

    # Put a legend to the right of the current axis
    ax.legend(title='',
              loc='upper left', ncol=1, bbox_to_anchor=(1.02, 1.0),
              frameon=0, prop={'size': 7})

    # spl.format_fig(ax,
    #                figtitle='System Fragility: ' + config.SYSTEM_CLASS,
    #                x_lab='Peak Ground Acceleration (g)',
    #                y_lab='P($D_s$ > $d_s$)',
    #                x_scale=None,
    #                y_scale=None,
    #                x_tick_val=None,
    #                y_tick_pos=y_tick_pos,
    #                y_tick_val=y_tick_val,
    #                x_grid=True,
    #                y_grid=True,
    #                add_legend=True)

    plt.savefig(outfig, format='png', dpi=300)
    plt.close(fig)

# ==============================================================================

def correct_crossover(SYS_DS, pb_exceed, x_sample, sys_dmg_fitted_params,
                      CROSSOVER_THRESHOLD=0.001):
    print(Fore.GREEN + "Checking for crossover ..." + Fore.RESET)
    params_pe = lmfit.Parameters()
    for dx in range(1, len(SYS_DS)):
        x_sample = x_sample
        y_sample = pb_exceed[dx]

        mu_hi = sys_dmg_fitted_params[dx].params['median'].value
        sd_hi = sys_dmg_fitted_params[dx].params['logstd'].value
        loc_hi = sys_dmg_fitted_params[dx].params['loc'].value

        y_model_hi = stats.lognorm.cdf(x_sample, sd_hi,
                                       loc=loc_hi, scale=mu_hi)

        params_pe.add('median', value=mu_hi, min=0, max=10)
        params_pe.add('logstd', value=sd_hi)
        params_pe.add('loc', value=0.0, vary=False)
        sys_dmg_fitted_params[dx] = lmfit.minimize(res_lognorm_cdf, params_pe,
                                         args=(x_sample, y_sample))

        ####################################################################
        if dx >= 2:
            mu_lo = sys_dmg_fitted_params[dx-1].params['median'].value
            sd_lo = sys_dmg_fitted_params[dx-1].params['logstd'].value
            loc_lo = sys_dmg_fitted_params[dx-1].params['loc'].value
            chi = sys_dmg_fitted_params[dx-1].chisqr
            y_model_lo = stats.lognorm.cdf(x_sample, sd_lo,
                                           loc=loc_lo, scale=mu_lo)

            # if sum(y_model_lo - y_model_hi < 0):
            if abs(min(y_model_lo - y_model_hi)) > CROSSOVER_THRESHOLD:
                print(Fore.MAGENTA
                      + "There is overlap for curve pair : "
                      + SYS_DS[dx - 1] + '-' + SYS_DS[dx]
                      + Fore.RESET)

                # Test if higher curve is co-incident with,
                # or precedes lower curve
                if (mu_hi <= mu_lo) or (loc_hi < loc_lo):
                    print(" *** Mean of higher curve too low: resampling")
                    params_pe.add('median', value=mu_hi, min=mu_lo)
                    sys_dmg_fitted_params[dx] = lmfit.minimize(
                        res_lognorm_cdf,
                        params_pe,
                        args=(x_sample, y_sample))

                    (mu_hi, sd_hi, loc_hi) = \
                        (sys_dmg_fitted_params[dx].params['median'].value,
                         sys_dmg_fitted_params[dx].params['logstd'].value,
                         sys_dmg_fitted_params[dx].params['loc'].value)

                # Thresholds for testing top or bottom crossover
                delta_top = (3.0 * sd_lo - (mu_hi - mu_lo)) / 3
                delta_btm = (3.0 * sd_lo + (mu_hi - mu_lo)) / 3

                # Test for top crossover: resample if crossover detected
                if (sd_hi < sd_lo) and (sd_hi <= delta_top):
                    print(" *** Attempting to correct upper crossover")
                    params_pe.add('logstd', value=sd_hi, min=delta_top)
                    sys_dmg_fitted_params[dx] = lmfit.minimize(
                        res_lognorm_cdf,
                        params_pe,
                        args=(x_sample, y_sample))

                # Test for bottom crossover: resample if crossover detected
                # elif (sd_hi >= sd_lo) and sd_hi >= delta_btm:
                elif sd_hi >= delta_btm:
                    print(" *** Attempting to correct lower crossover")
                    params_pe.add('logstd', value=sd_hi, max=delta_btm)
                    sys_dmg_fitted_params[dx] = lmfit.minimize(
                        res_lognorm_cdf,
                        params_pe,
                        args=(x_sample, y_sample))

            else:
                print(Fore.GREEN
                      + "There is NO overlap for given THRESHOLD of "
                      + str(CROSSOVER_THRESHOLD)
                      + ", for curve pair: "
                      + SYS_DS[dx - 1] + '-' + SYS_DS[dx]
                      + Fore.RESET)

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


def res_norm_cdf(params, x, data, eps=None):
    mu = params['mean'].value
    sd = params['stddev'].value
    model = stats.norm.cdf(x, loc=mu, scale=sd)
    if eps is None:
        return (model - data)
    return (model - data) / eps


def bimodal_norm_cdf(x, m1, s1, w1, m2, s2, w2):
    return w1 * norm_cdf(x, m1, s1) + w2 * norm_cdf(x, m2, s2)


def res_bimodal_norm_cdf(params, x, data, eps=None):
    m1 = params['m1'].value
    s1 = params['s1'].value
    w1 = params['w1'].value
    m2 = params['m2'].value
    s2 = params['s2'].value
    w2 = params['w2'].value
    model = bimodal_norm_cdf(x, m1, s1, w1, m2, s2, w2)
    if eps is None:
        return (model - data)
    return (model - data) / eps


# ============================================================================

def fit_restoration_data(RESTORATION_TIME_RANGE, sys_fn, SYS_DS, out_path):
    """
    Fits a normal CDF to each of the damage states, i.e. for each column of
    data in 'sys_fn'

    :param RESTORATION_TIME_RANGE: restoration time range (numpy array)
    :param sys_fn: system functionality restoration over time (2D numpy array)
    :param SYS_DS: discrete damage states (list)
    :param out_path: directory path for writing output (string)
    :returns:  fitted restoration model parameters (PANDAS dataframe)
    """

    indx = pd.Index(SYS_DS[1:], name='Damage States')
    sys_rst_mdl_mode1 = pd.DataFrame(index=indx,
                                     columns=['Mean',
                                              'StdDev',
                                              'Chi-Sqr'])

    # ----- Get the initial fit -----
    # sys_rst_ci = [{} for _ in xrange(len(SYS_DS))]

    sys_rst_fit = [[] for _ in xrange(len(SYS_DS))]
    for dx in range(1, len(SYS_DS)):
        x_sample = RESTORATION_TIME_RANGE
        y_sample = sys_fn[SYS_DS[dx]]

        # Fit the dist. Add initial estimate if needed.
        init_m = np.mean(y_sample)
        init_s = np.std(y_sample)

        params = lmfit.Parameters()
        params.add('mean', value=init_m)
        params.add('stddev', value=init_s)

        sys_rst_fit[dx] = lmfit.minimize(res_norm_cdf, params,
                                         args=(x_sample, y_sample),
                                         method='leastsq')

        sys_rst_mdl_mode1.loc[SYS_DS[dx]] \
            = sys_rst_fit[dx].params['mean'].value, \
              sys_rst_fit[dx].params['stddev'].value, \
              sys_rst_fit[dx].chisqr

    print("\n\n" + "-" * 79)
    print(Fore.YELLOW +
          "Fitting system RESTORATION data: Unimodal Normal CDF" +
          Fore.RESET)
    print("-" * 79)

    # # Format output to limit displayed decimal precision
    # sys_rst_mdl_mode1['Mean'] = \
    #     sys_rst_mdl_mode1['Mean'].map('{:,.3f}'.format)
    # sys_rst_mdl_mode1['StdDev'] = \
    #     sys_rst_mdl_mode1['StdDev'].map('{:,.3f}'.format)

    print("INITIAL Restoration Parameters:\n\n", sys_rst_mdl_mode1, '\n')

    # ----- Check for crossover and resample as needed -----
    for dx in range(1, len(SYS_DS)):
        x_sample = RESTORATION_TIME_RANGE
        y_sample = sys_fn[SYS_DS[dx]]

        m1_hi = sys_rst_fit[dx].params['mean'].value
        s1_hi = sys_rst_fit[dx].params['stddev'].value
        y_model_hi = norm_cdf(x_sample, m1_hi, s1_hi)

        # --------------------------------------------------------------------
        # Check for crossover...

        if dx >= 2:
            m1_lo, s1_lo, r1_chi = sys_rst_mdl_mode1.loc[SYS_DS[dx - 1]].values
            y_model_lo = norm_cdf(x_sample, m1_lo, s1_lo)

            if sum(y_model_lo - y_model_hi < 0):
                print(Fore.MAGENTA +
                      "There is overlap for curve pair   : " +
                      SYS_DS[dx - 1] + '-' + SYS_DS[dx] +
                      Fore.RESET)

                k = 0
                crossover = True
                mu_err = 0
                sdtop_err = 0
                sdbtm_err = 0
                while k < 50 and crossover:
                    # Test if higher curve is co-incident with,
                    #   or precedes lower curve
                    if (m1_hi <= m1_lo):
                        if not mu_err > 0:
                            print("   *** Attempting to correct mean...")
                        params.add('mean', value=m1_hi, min=m1_lo * 1.01)
                        sys_rst_fit[dx] = lmfit.minimize(
                            res_norm_cdf, params, args=(x_sample, y_sample))

                        (m1_hi, s1_hi) = \
                            (sys_rst_fit[dx].params['mean'].value,
                             sys_rst_fit[dx].params['stddev'].value)
                        mu_err += 1

                    # Thresholds for testing top or bottom crossover
                    delta_top = (1 + k / 100.0) * (
                    3.0 * s1_lo - (m1_hi - m1_lo)) / 3
                    delta_btm = (1 - k / 100.0) * (
                    3.0 * s1_lo + (m1_hi - m1_lo)) / 3

                    # Test for top crossover: resample if x-over detected
                    if (s1_hi < s1_lo) or (s1_hi <= delta_top):
                        if not sdtop_err > 0:
                            print("   *** " +
                                  "Attempting to correct top crossover...")
                        params.add('mean', value=m1_hi * 1.01,
                                   min=m1_lo * 1.01)
                        params.add('stddev', value=s1_hi, min=delta_top)
                        sys_rst_fit[dx] = lmfit.minimize(
                            res_norm_cdf, params, args=(x_sample, y_sample))

                        (m1_hi, s1_hi) = \
                            (sys_rst_fit[dx].params['mean'].value,
                             sys_rst_fit[dx].params['stddev'].value)
                        sdtop_err += 1

                    # Test for bottom crossover: resample if x-over detected
                    elif (s1_hi >= delta_btm):
                        if not sdbtm_err > 0:
                            print("   *** " +
                                  "Attempting to correct bottom crossover...")
                        params.add('stddev', value=s1_hi, min=delta_btm)
                        sys_rst_fit[dx] = lmfit.minimize(
                            res_norm_cdf, params, args=(x_sample, y_sample))

                        (m1_hi, s1_hi) = \
                            (sys_rst_fit[dx].params['mean'].value,
                             sys_rst_fit[dx].params['stddev'].value)
                        sdbtm_err += 1

                    y_model_hi = norm_cdf(x_sample, m1_hi, s1_hi)
                    crossover = sum(y_model_lo < y_model_hi)
                    k += 1

                # Test if crossover correction succeeded
                if not sum(y_model_lo < y_model_hi):
                    print(Fore.YELLOW +
                          "   Crossover corrected!" +
                          Fore.RESET)
                else:
                    print(Fore.RED + Style.BRIGHT +
                          "   Crossover NOT corrected!" +
                          Fore.RESET + Style.RESET_ALL)

            else:
                print(Fore.GREEN + "There is NO overlap for curve pair: " +
                      SYS_DS[dx - 1] + '-' + SYS_DS[dx] +
                      Fore.RESET)

        # --------------------------------------------------------------------
        # * Need to find a solution to reporting confidence interval reliably:
        #
        # sys_rst_ci[dx], trace = lmfit.conf_interval(sys_rst_fit[dx], \
        #                     sigmas=[0.674,0.950,0.997], trace=True)
        # --------------------------------------------------------------------

        sys_rst_mdl_mode1.loc[SYS_DS[dx]] \
            = sys_rst_fit[dx].params['mean'].value, \
              sys_rst_fit[dx].params['stddev'].value, \
              sys_rst_fit[dx].chisqr

    # sys_rst_mdl_mode1['Mean'] = \
    #     sys_rst_mdl_mode1['Mean'].map('{:,.3f}'.format)
    # sys_rst_mdl_mode1['StdDev'] = \
    #     sys_rst_mdl_mode1['StdDev'].map('{:,.3f}'.format)

    print("\nFINAL Restoration Parameters: \n")
    print(sys_rst_mdl_mode1)

    # for dx in range(1, len(SYS_DS)):
    #     print("\n\nRestoration model statistics for damage state: %s"
    #           % SYS_DS[dx])
    #     print("Goodness-of-Fit chi-square test statistic: %f"
    #           % sys_rst_fit[dx].chisqr)
    #     print("Confidence intervals: ")
    #     lmfit.printfuncs.report_ci(sys_rst_ci[dx])

    sys_rst_mdl_mode1.to_csv(os.path.join(out_path,
                                          'system_model_restoration__mode1.csv'),
                             sep=',')

    fig = plt.figure(figsize=(9, 4.5), facecolor='white')
    ax = fig.add_subplot(111, axisbg='white')

    # --- Plot simulation data points ---
    spl.add_legend_subtitle("Simulation Data:")
    for i in range(1, len(SYS_DS)):
        ax.plot(RESTORATION_TIME_RANGE[1:],
                sys_fn[SYS_DS[i]][1:] * 100,
                label=SYS_DS[i], clip_on=False,
                color=spl.COLR_DS[i], linestyle='', alpha=0.35,
                marker=markers[i - 1], markersize=4, markeredgecolor=set2[-i])

    # --- Plot the fitted models ---
    spl.add_legend_subtitle("\nModel: Normal CDF")
    for dx in range(1, len(SYS_DS)):
        m1 = sys_rst_mdl_mode1.loc[SYS_DS[dx]]['Mean']
        s1 = sys_rst_mdl_mode1.loc[SYS_DS[dx]]['StdDev']
        ax.plot(RESTORATION_TIME_RANGE[1:],
                norm_cdf(RESTORATION_TIME_RANGE, m1, s1)[1:] * 100,
                label=SYS_DS[dx], clip_on=False, color=spl.COLR_DS[dx],
                linestyle='-', linewidth=1.5, alpha=0.65)

    x_pwr = int(np.ceil(np.log10(max(RESTORATION_TIME_RANGE))))
    x_tiks = [10 ** t for t in range(0, x_pwr + 1)]

    outfig = os.path.join(out_path, 'fig_MODEL_sys_rst_mode1.png')
    ax.margins(0.03, None)
    spl.format_fig(ax,
                   figtitle='Restoration Model for: ' + fc.system_class,
                   x_lab='Time (' + sc.time_unit + ')',
                   y_lab='Percent Functional',
                   x_scale='log',  # <OR> None
                   y_scale=None,
                   x_tick_pos=x_tiks,
                   x_tick_val=x_tiks,
                   y_tick_val=range(0, 101, 20),
                   x_lim=[min(x_tiks), max(x_tiks)],
                   y_lim=[0, 100],
                   x_grid=True,
                   y_grid=True,
                   add_legend=True)

    plt.savefig(outfig, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    return sys_rst_mdl_mode1


###############################################################################


# sturges = lambda n: int(np.log2(n) + 1)
# sys_fn[DS].hist(bins=sturges(sys_fn[DS].size), normed=True,
#     color='lightseagreen')
# sys_fn[DS].dropna().plot(kind='kde', xlim=(0,100), style='r--')
# plt.show(block=False)

def fit_restoration_data_multimode(RESTORATION_TIME_RANGE,
                                   sys_fn, SYS_DS, out_path):

    """
    *********************************************************************
    This function is not yet mature and is meant only for experimentation
    *********************************************************************

    Function for fitting a bimodal normal cdf to restoration data

    :param RESTORATION_TIME_RANGE: restoration time range (numpy array)
    :param sys_fn: system functionality restoration over time (2D numpy array)
    :param SYS_DS: discrete damage states (list)
    :param out_path: directory path for writing output (string)
    :returns:  fitted restoration model parameters (PANDAS dataframe)
    """
    indx = pd.Index(SYS_DS[1:], name='Damage States')
    sys_rst_mdl_mode2 = pd.DataFrame(index=indx,
                                     columns=['Mean1', 'SD1', 'Weight1',
                                              'Mean2', 'SD2', 'Weight2',
                                              'Chi-Sqr'])

    sys_mix_fit = [[] for _ in xrange(len(SYS_DS))]

    for dx in range(1, len(SYS_DS)):
        DS = SYS_DS[dx]

        x_sample = RESTORATION_TIME_RANGE
        y_sample = sys_fn[DS]
        (m_est, s_est), pcov = curve_fit(norm_cdf, x_sample, y_sample)

        params_mx = lmfit.Parameters()
        params_mx.add('m1', value=m_est)
        params_mx.add('s1', value=s_est)
        params_mx.add('w1', value=0.6)
        params_mx.add('m2', value=m_est)
        params_mx.add('s2', value=s_est)
        params_mx.add('w2', value=0.4)

        sys_mix_fit[dx] = lmfit.minimize(res_bimodal_norm_cdf, params_mx,
                                         args=(x_sample, y_sample),
                                         method='leastsq')

        m1 = sys_mix_fit[dx].params['m1'].value
        s1 = sys_mix_fit[dx].params['s1'].value
        w1 = sys_mix_fit[dx].params['w1'].value
        m2 = sys_mix_fit[dx].params['m2'].value
        s2 = sys_mix_fit[dx].params['s2'].value
        w2 = sys_mix_fit[dx].params['w2'].value

        # sys_mix_ci[dx] = lmfit.conf_interval(sys_mix_fit[dx], \
        #                     sigmas=[0.674,0.950,0.997], trace=False)

        sys_rst_mdl_mode2.loc[DS] = m1, s1, w1, m2, s2, w2, \
                                   sys_mix_fit[dx].chisqr

    sys_rst_mdl_mode2.to_csv(os.path.join(sc.output_path,
                                          'system_model_restoration__mode2.csv'),
                             sep=',')

    print("\n\n" + "-" * 79)
    print("System Restoration Parameters: Bimodal Normal CDF Model")
    print("-" * 79 + "\n")
    print(sys_rst_mdl_mode2)

    # sys_rst_ci_df = ci_dict_to_df(sys_mix_ci)
    # print("Confidence intervals: ")
    # lmfit.printfuncs.report_ci(sys_mix_ci[dx])

    # ........................................................................

    # w, h = plt.figaspect(0.5)
    w, h = [9, 4.5]
    fig = plt.figure(figsize=(w, h), dpi=250, facecolor='white')
    ax = fig.add_subplot(111, axisbg='white')

    spl.add_legend_subtitle("Simulation Data")
    for dx in range(1, len(SYS_DS)):
        DS = SYS_DS[dx]
        x_sample = RESTORATION_TIME_RANGE
        plt.plot(
            x_sample[1:],
            sys_fn[DS].values[1:] * 100,
            label=DS, clip_on=False, color=spl.COLR_DS[dx], alpha=0.4,
            linestyle='', marker=markers[dx - 1], markersize=4
        )

    spl.add_legend_subtitle("\nModel: Bimodal Normal CDF")
    for dx in range(1, len(SYS_DS)):
        DS = SYS_DS[dx]
        x_sample = RESTORATION_TIME_RANGE
        plt.plot(
            x_sample[1:],
            bimodal_norm_cdf(
                x_sample, *sys_rst_mdl_mode2.loc[DS].values[:-1])[1:] * 100,
            label=DS, clip_on=False, color=spl.COLR_DS[dx], alpha=0.65,
            linestyle='-', linewidth=1.5
        )

    x_pwr = int(np.ceil(np.log10(max(RESTORATION_TIME_RANGE))))
    x_tiks = [10 ** t for t in range(0, x_pwr + 1)]

    outfig = os.path.join(out_path, 'fig_MODEL_sys_rst_mode2.png')
    ax.margins(0.03, None)
    spl.format_fig(ax,
                   figtitle='Multimodal Restoration Model for: ' +
                            fc.system_class,
                   x_lab='Time (' + sc.time_unit + ')',
                   y_lab='Percent Functional',
                   x_scale='log',
                   y_scale=None,
                   x_tick_pos=x_tiks,
                   x_tick_val=x_tiks,
                   y_tick_val=range(0, 101, 20),
                   x_lim=[min(x_tiks), max(x_tiks)],
                   y_lim=[0, 100],
                   x_grid=True,
                   y_grid=True,
                   add_legend=True)

    plt.savefig(outfig, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    return sys_rst_mdl_mode2


# ============================================================================
# Calculate SYSTEM RESTORATION over time, given damage state
# ----------------------------------------------------------------------------

def approximate_generic_sys_restoration(sc, fc, sys_frag, output_array_given_recovery):
    SYS_DS = fc.sys_dmg_states
    sys_fn = pd.DataFrame(index=sc.restoration_time_range,
                          columns=[fc.sys_dmg_states])
    sys_fn.fillna(1)
    sys_fn.index.name = "Time in " + sc.time_unit

    for ds in range(len(SYS_DS)):
        fn_tmp = np.zeros((sc.num_hazard_pts, sc.num_time_steps))
        ids = {}  # index of damage states within the samples
        for p in range(sc.num_hazard_pts):
            ids[p] = np.where(sys_frag[:, p] == ds)[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                m = np.mean(output_array_given_recovery[ids[p], p, :], axis=0)
            fn_tmp[p] = m / fc.nominal_production
        sys_fn[SYS_DS[ds]] = np.nanmean(fn_tmp, axis=0)

    # sys_fn = sys_fn.drop('DS0 None', axis=1)
    sys_fn.to_csv(os.path.join(
        sc.output_path, 'system_restoration_profile.csv'), sep=',')

    return sys_fn

