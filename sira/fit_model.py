from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range

import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import sira.siraplot as spl

import numpy as np
from scipy import stats
import lmfit
import pandas as pd
pd.options.display.float_format = '{:,.4f}'.format

import os

import brewer2mpl
from colorama import Fore, init
init()

import logging
rootLogger = logging.getLogger(__name__)

MIN = 0
MAX = 10


# ----------------------------------------------------------------------------
# For plots: using the  brewer2 color maps by Cynthia Brewer
# ----------------------------------------------------------------------------

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

def lognormal_cdf(x, median, beta, loc=0):
    scale = median
    shape = beta
    loc = loc
    return stats.lognorm.cdf(x, shape, loc=loc, scale=scale)


def res_lognorm_cdf(params, x, data, eps=None):
    v = params.valuesdict()
    model = stats.lognorm.cdf(x, v['beta'], loc=v['loc'], scale=v['median'])
    if eps is None:
        return (model - data)
    return (model - data) / eps


def fit_prob_exceed_model(
        hazard_input_vals,
        pb_exceed,
        SYS_DS,
        out_path,
        config):
    """
    Fit a Lognormal CDF model to simulated probability exceedance data

    :param hazard_input_vals: input values for hazard intensity (numpy array)
    :param pb_exceed:   probability of exceedance (2D numpy array)
                        its shape is (num_damage_states x num_hazards_points)
    :param SYS_DS: discrete damage states (list of strings)
    :param out_path: directory path for writing output (string)
    :returns:  fitted exceedance model parameters (PANDAS dataframe)
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # DataFrame for storing the calculated System Damage Algorithms
    # for exceedence probabilities.
    indx = pd.Index(SYS_DS[1:], name='Damage States')
    sys_dmg_model = pd.DataFrame(
        index=indx,
        columns=['Median', 'LogStdDev', 'Location', 'Chi-Sqr'])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # INITIAL FIT

    sys_dmg_fitted_params = [[] for _ in range(len(SYS_DS))]
    hazard_input_vals = [float(x) for x in hazard_input_vals]

    for dx in range(1, len(SYS_DS)):

        x_sample = hazard_input_vals
        y_sample = pb_exceed[dx]

        p0m = np.mean(y_sample)
        p0s = np.std(y_sample)

        if dx >= 2 and p0m < sys_dmg_fitted_params[dx-1].params['median'].value:
            p0m = sys_dmg_fitted_params[dx-1].params['median'].value+0.02

        # Fit the dist:
        params_est = lmfit.Parameters()
        params_est.add('median', value=p0m, min=MIN, max=MAX)
        params_est.add('beta', value=p0s, min=MIN, max=MAX)
        params_est.add('loc', value=0.0, vary=False)

        sys_dmg_fitted_params[dx] = lmfit.minimize(
            res_lognorm_cdf,
            params_est,
            args=(x_sample, y_sample),
            method='leastsq',
            nan_policy='omit',
            maxfev=1000
            )
        sys_dmg_model.loc[SYS_DS[dx]] = (
            sys_dmg_fitted_params[dx].params['median'].value,
            sys_dmg_fitted_params[dx].params['beta'].value,
            sys_dmg_fitted_params[dx].params['loc'].value,
            sys_dmg_fitted_params[dx].chisqr
            )

    border = "-" * 79
    rootLogger.info(
        '\n'+border+'\n'+
        Fore.YELLOW+"Fitting system FRAGILITY data: Lognormal CDF"+Fore.RESET+
        '\n' + border + '\n' +
        "\nINITIAL System Fragilities:\n\n" +
        str(sys_dmg_model) + '\n'
        )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Check for crossover and correct as needed

    CROSSOVER_THRESHOLD = 0.005
    CROSSOVER_CORRECTION = True
    if CROSSOVER_CORRECTION:
        sys_dmg_fitted_params = correct_crossover(
            SYS_DS,
            pb_exceed,
            hazard_input_vals,
            sys_dmg_fitted_params,
            CROSSOVER_THRESHOLD)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Finalise damage function parameters

    for dx in range(1, len(SYS_DS)):
        sys_dmg_model.loc[SYS_DS[dx]] = \
            sys_dmg_fitted_params[dx].params['median'].value, \
            sys_dmg_fitted_params[dx].params['beta'].value, \
            sys_dmg_fitted_params[dx].params['loc'].value, \
            sys_dmg_fitted_params[dx].chisqr

    rootLogger.info("\n\nFINAL System Fragilities:\n\n" +
                    str(sys_dmg_model) + '\n')

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
                    file_name='fig_sys_pe_DATA.png',
                    PLOT_DATA=True,
                    PLOT_MODEL=False,
                    PLOT_EVENTS=False)

    plot_data_model(SYS_DS,
                    hazard_input_vals,
                    sys_dmg_model,
                    pb_exceed,
                    out_path,
                    config,
                    file_name='fig_sys_pe_MODEL.png',
                    PLOT_DATA=False,
                    PLOT_MODEL=True,
                    PLOT_EVENTS=False)

    plot_data_model(SYS_DS,
                    hazard_input_vals,
                    sys_dmg_model,
                    pb_exceed,
                    out_path,
                    config,
                    file_name='fig_sys_pe_MODEL_with_scenarios.png',
                    PLOT_DATA=False,
                    PLOT_MODEL=True,
                    PLOT_EVENTS=True)

    # ------------------------------------------------------------
    # RETURN a DataFrame with the fitted model parameters
    return sys_dmg_model

# ==============================================================================

def plot_data_model(SYS_DS,
                    hazard_input_vals,
                    sys_dmg_model,
                    pb_exceed,
                    out_path,
                    config,
                    file_name='fig_sys_pe.png',
                    PLOT_DATA=True,
                    PLOT_MODEL=True,
                    PLOT_EVENTS=False):

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if sum([PLOT_DATA, PLOT_MODEL, PLOT_EVENTS])==0:
        raise AttributeError

    plt.style.use('seaborn-darkgrid')
    mpl.rc('grid', linewidth=0.7)
    mpl.rc('font', family='sans-serif')
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    colours = spl.ColourPalettes()
    COLR_DS = colours.FiveLevels[-1*len(SYS_DS):]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # [Plot 1 of 3] The Data Points

    if PLOT_DATA:
        spl.add_legend_subtitle("DATA")
        for i in range(1, len(SYS_DS)):
            ax.plot(hazard_input_vals, pb_exceed[i],
                    label=SYS_DS[i], clip_on=False, color=COLR_DS[i],
                    linestyle='', alpha=0.6, marker=markers[i-1],
                    markersize=3, markeredgewidth=1, markeredgecolor=None,
                    zorder=10)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # [Plot 2 of 3] The Fitted Model

    if PLOT_MODEL:
        spl.add_legend_subtitle("FITTED MODEL")
        xmax = max(hazard_input_vals)
        xformodel = np.linspace(0, xmax, 101, endpoint=True)
        dmg_mdl_arr = np.zeros((len(SYS_DS), len(xformodel)))

        for dx in range(1, len(SYS_DS)):
            shape = sys_dmg_model.loc[SYS_DS[dx], 'LogStdDev']
            loc = sys_dmg_model.loc[SYS_DS[dx], 'Location']
            scale = sys_dmg_model.loc[SYS_DS[dx], 'Median']
            dmg_mdl_arr[dx] = stats.lognorm.cdf(
                xformodel, shape, loc=loc, scale=scale)
            ax.plot(xformodel, dmg_mdl_arr[dx],
                    label=SYS_DS[dx], clip_on=False, color=COLR_DS[dx],
                    alpha=0.65, linestyle='-', linewidth=1.6,
                    zorder=9)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # [Plot 3 of 3] The Scenario Events

    if PLOT_EVENTS:
        spl.add_legend_subtitle("EVENTS")
        for i, haz in enumerate(config.FOCAL_HAZARD_SCENARIOS):
            event_num = str(i+1)
            event_intensity_str = "{:.3f}".format(float(haz))
            event_color = colours.BrewerSpectral[i]
            try:
                event_label = event_num + ". " + \
                              config.FOCAL_HAZARD_SCENARIO_NAMES[i]+" : " + \
                              event_intensity_str
            except:
                event_label = event_num + " : " + event_intensity_str

            ax.plot(float(haz), 0,
                    label=event_label,
                    color=event_color,
                    marker='',
                    markersize=2,
                    linestyle='-',
                    zorder=11)
            ax.plot(float(haz), 1.04,
                    label='',
                    clip_on=False,
                    color=event_color,
                    marker='o',
                    fillstyle='none',
                    markersize=12,
                    linestyle='-',
                    markeredgewidth=1.0,
                    zorder=11)
            ax.annotate(
                event_num,  # event_intensity_str,
                xy=(float(haz), 0), xycoords='data',
                xytext=(float(haz), 1.038), textcoords='data',
                ha='center', va='center', rotation=0,
                size=8, fontweight='bold', 
                color=event_color,
                annotation_clip=False,
                bbox=dict(boxstyle='round, pad=0.2', fc='yellow', alpha=0.0),
                path_effects=[
                    PathEffects.withStroke(linewidth=2, foreground="w")],
                arrowprops=dict(
                    arrowstyle='-|>, head_length=0.5, head_width=0.3',
                    shrinkA=3.0,
                    shrinkB=0.0,
                    connectionstyle='arc3,rad=0.0',
                    color=event_color,
                    alpha=0.8,
                    linewidth=1.0,
                    linestyle="-",
                    path_effects=[
                        PathEffects.withStroke(linewidth=2.5, foreground="w")
                        ]
                    ),
                zorder=11
                )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ax.set_axisbelow('line')

    outfig = os.path.join(out_path, file_name)
    figtitle = 'System Fragility: ' + config.MODEL_NAME

    x_lab = config.HAZARD_INTENSITY_MEASURE_PARAM + \
            ' (' + config.HAZARD_INTENSITY_MEASURE_UNIT + ')'
    y_lab = 'P($D_s$ > $d_s$)'

    y_tick_pos = np.linspace(0.0, 1.0, num=6, endpoint=True)
    y_tick_val = ['{:.1f}'.format(i) for i in y_tick_pos]
    x_tick_pos = np.linspace(0.0, max(hazard_input_vals), num=6, endpoint=True)
    x_tick_val = ['{:.2f}'.format(i) for i in x_tick_pos]

    ax.set_title(figtitle, loc='center', y=1.06, fontweight='bold', size=11)
    ax.set_xlabel(x_lab, size=10, labelpad=10)
    ax.set_ylabel(y_lab, size=10, labelpad=10)

    ax.set_xlim(0, max(x_tick_pos))
    ax.set_xticks(x_tick_pos)
    ax.set_xticklabels(x_tick_val, size=9)
    ax.set_ylim(0, max(y_tick_pos))
    ax.set_yticks(y_tick_pos)
    ax.set_yticklabels(y_tick_val, size=9)
    ax.margins(0, 0)
    ax.tick_params(axis='both', pad=7)

    # Shrink current axis width by 15%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    # Put a legend to the right of the current axis
    ax.legend(title='', loc='upper left', ncol=1,
              bbox_to_anchor=(1.02, 1.0), frameon=0, prop={'size': 9})

    plt.savefig(outfig, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==============================================================================

def correct_crossover(
        SYS_DS,
        pb_exceed,
        x_data,
        sys_dmg_fitted_params,
        CROSSOVER_THRESHOLD=0.005):

    rootLogger.info(Fore.GREEN +
                    "Checking for crossover [ THRESHOLD = {} ]".
                    format(str(CROSSOVER_THRESHOLD)) + Fore.RESET)
    params_pe = lmfit.Parameters()
    for dx in range(1, len(SYS_DS)):
        x_sample = x_data
        y_sample = pb_exceed[dx]

        mu_hi = sys_dmg_fitted_params[dx].params['median'].value
        sd_hi = sys_dmg_fitted_params[dx].params['beta'].value
        loc_hi = sys_dmg_fitted_params[dx].params['loc'].value

        y_model_hi = stats.lognorm.cdf(x_sample, sd_hi, loc=loc_hi, scale=mu_hi)

        params_pe.add('median', value=mu_hi, min=MIN, max=MAX)
        params_pe.add('beta', value=sd_hi, min=0, max=MAX)
        params_pe.add('loc', value=0.0, vary=False)
        sys_dmg_fitted_params[dx] = lmfit.minimize(
            res_lognorm_cdf,
            params_pe,
            args=(x_sample, y_sample),
            method='leastsq',
            nan_policy='omit',
            )

        ####################################################################
        if dx >= 2:
            mu_lo = sys_dmg_fitted_params[dx-1].params['median'].value
            sd_lo = sys_dmg_fitted_params[dx-1].params['beta'].value
            loc_lo = sys_dmg_fitted_params[dx-1].params['loc'].value
            y_model_lo = stats.lognorm.cdf(
                x_sample, sd_lo, loc=loc_lo, scale=mu_lo)

            if abs(min(y_model_lo - y_model_hi)) > CROSSOVER_THRESHOLD:
                rootLogger.info(
                    Fore.MAGENTA +
                    "There is overlap for curve pair : " +
                    SYS_DS[dx - 1] + '-' + SYS_DS[dx] +
                    Fore.RESET)

                # Test if higher curve is co-incident with, or
                # precedes lower curve
                if (mu_hi <= mu_lo) or (loc_hi < loc_lo):
                    rootLogger.info(" *** Mean of higher curve too low: "
                                    "resampling")
                    rootLogger.info('median '+str(mu_hi)+" "+str(mu_lo))
                    params_pe.add('median', value=mu_hi, min=mu_lo)
                    sys_dmg_fitted_params[dx] = lmfit.minimize(
                        res_lognorm_cdf, params_pe, args=(x_sample, y_sample))

                    (mu_hi, sd_hi, loc_hi) = (
                        sys_dmg_fitted_params[dx].params['median'].value,
                        sys_dmg_fitted_params[dx].params['beta'].value,
                        sys_dmg_fitted_params[dx].params['loc'].value)

                # Thresholds for testing top or bottom crossover
                delta_top = sd_lo - (mu_hi - mu_lo)/1.0
                delta_btm = sd_lo + (mu_hi - mu_lo)/1.0

                # Test for top crossover: resample if crossover detected
                if (sd_hi < sd_lo) and (sd_hi <= delta_top):
                    rootLogger.info("*** Attempting to correct upper crossover")
                    params_pe.add('beta', value=sd_hi, min=delta_top)
                    sys_dmg_fitted_params[dx] = lmfit.minimize(
                        res_lognorm_cdf, params_pe, args=(x_sample, y_sample))

                # Test for bottom crossover: resample if crossover detected
                elif sd_hi >= delta_btm:
                    rootLogger.info("*** Attempting to correct lower crossover")
                    params_pe.add('beta', value=sd_hi, max=delta_btm)
                    sys_dmg_fitted_params[dx] = lmfit.minimize(
                        res_lognorm_cdf, params_pe, args=(x_sample, y_sample))

            else:
                rootLogger.info(
                    Fore.GREEN +
                    "There is NO overlap for curve pair: " +
                    SYS_DS[dx - 1] + '-' + SYS_DS[dx] + Fore.RESET)

    return sys_dmg_fitted_params

# ==============================================================================
#
# NORMAL CURVE FITTING
#
# ------------------------------------------------------------------------------
# Parameters in scipy NORMAL distribution:
#
# The location (loc) keyword specifies the mean.
# The scale (scale) keyword specifies the standard deviation.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
#
# Note on the covariance matrix returned by scipy.optimize.curve_fit:
# The square root of the diagonal values are the 1-sigma uncertainties of
# the fit parameters
# ------------------------------------------------------------------------------

def norm_cdf(x, mu, sd):
    return stats.norm.cdf(x, loc=mu, scale=sd)
