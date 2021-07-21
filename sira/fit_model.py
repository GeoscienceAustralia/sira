import sys
import warnings
import logging
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erf  # noqa:E0611
import lmfit

import brewer2mpl
from colorama import Fore, init

import sira.tools.siraplot as spl

init()
rootLogger = logging.getLogger(__name__)
mpl.use('agg')
pd.options.display.float_format = '{:,.4f}'.format

# -----------------------------------------------------------------------------
# For plots: using the  brewer2 color maps by Cynthia Brewer
# -----------------------------------------------------------------------------

set2 = brewer2mpl.get_map('Set2', 'qualitative', 5).mpl_colors
markers = ['o', '^', 's', 'D', 'x', '+']

class color:
    """From: https://stackoverflow.com/a/17303428"""
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# =============================================================================
#
# PROBABILITY of EXCEEDANCE MODEL FITTING
#
# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


def lognormal_cdf(x, median, beta, loc=0):
    x = np.asarray(x)
    logn_cdf = 0.5 + 0.5 * erf(
        (np.log(x - loc) - np.log(median)) / (np.sqrt(2) * beta))
    return logn_cdf


def normal_cdf(x, mean, stddev):
    return 0.5 * (1 + erf((x - mean) / (stddev * np.sqrt(2))))


def rayleigh_cdf(x, loc, scale):
    return stats.rayleigh.cdf(x, loc=loc, scale=scale)


def res_lognorm_cdf(params, x, data, eps=None):
    v = params.valuesdict()
    median = v['median']
    beta = v['beta']
    loc = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = lognormal_cdf(x, median, beta, loc=loc)
    if eps is None:
        return (model - data)
    return (model - data) / eps


def res_norm_cdf(params, x, data, eps=None):
    v = params.valuesdict()
    model = normal_cdf(x, v['mean'], v['stddev'])
    if eps is None:
        return (model - data)
    return (model - data) / eps


def res_rayleigh_cdf(params, x, data, eps=None):
    v = params.valuesdict()
    model = stats.rayleigh.cdf(x, loc=v['loc'], scale=v['scale'])
    if eps is None:
        return (model - data)
    return (model - data) / eps

# ====================================================================================


def display_dict(pdict, width=4, level=0, init_space=0):
    """Neatly prints a dict of params"""
    ip = init_space
    for key, value in pdict.items():
        if isinstance(value, float):
            val_fmt = f"{value:<.4f}"
        else:
            val_fmt = f"{str(value):<12}"
        if not(isinstance(value, dict) or isinstance(value, OrderedDict)) and (level < 1):
            print(' ' * init_space + '[' + str(key) + ']')
            print(f"{' ' * (init_space + width)}{val_fmt}")
        if isinstance(value, dict):
            print(' ' * (init_space + width * level) + '[' + str(key) + ']')
            display_dict(value, level=level + 1, init_space=ip)
        elif level > 0:
            print(f"{' '*(init_space + width*level)}{str(key):<10} = {val_fmt}")


def fit_cdf_lmfitmodel(x_sample, y_sample, dist):
    """
    Fits given array-like data using `lmfit.Model` method
    """

    # Locate x-values where the y-values are changing
    x_sample = np.asarray(x_sample)
    y_sample = np.asarray(y_sample)
    change_ndx = np.where(y_sample[:-1] != y_sample[1:])[0]
    change_ndx = list(change_ndx)

    if not (change_ndx[-1] == len(y_sample)):
        change_ndx.insert(len(change_ndx), change_ndx[-1] + 1)

    if change_ndx[0] != 0:
        change_ndx.insert(0, change_ndx[0] - 1)

    xs = x_sample[change_ndx]

    # -------------------------------------------------------------------------
    # NORMAL CDF -- set up model and params
    # -------------------------------------------------------------------------
    if dist.lower() in ["normal", "gaussian", "normal_cdf"]:
        func = normal_cdf
        model_dist = lmfit.Model(func)
        model_params = model_dist.make_params()
        model_params.add('mean', value=np.mean(xs), min=min(xs), max=np.mean(xs) * 2)
        model_params.add('stddev', value=np.std(xs), min=0, max=np.mean(xs) * 0.9)

    # -------------------------------------------------------------------------
    # LOGNORMAL CDF -- set up model and params
    # -------------------------------------------------------------------------
    elif dist.lower() in ["lognormal", "lognormal_cdf"]:
        func = lognormal_cdf
        model_dist = lmfit.Model(func)
        init_med = np.mean(xs)
        init_lsd = abs(np.std(xs))
        model_params = model_dist.make_params()
        model_params.add('median', value=init_med, min=min(xs), max=init_med * 2)
        model_params.add('beta', value=init_lsd, min=sys.float_info.min, max=init_med * 2)
        model_params.add('loc', value=0, vary=False)

    # -------------------------------------------------------------------------
    # RAYLEIGH CDF -- set up model and params
    # -------------------------------------------------------------------------
    elif dist.lower() in ["rayleigh", "rayleigh_cdf"]:
        func = rayleigh_cdf
        model_dist = lmfit.Model(func)
        init_loc = xs[0]
        init_scale = np.std(xs)
        model_params = model_dist.make_params()
        model_params.add('loc', value=init_loc, min=None, max=None)
        model_params.add('scale', value=init_scale, min=None, max=None)

    else:
        raise ValueError(f"The distribution {dist} is not supported.")

    # -------------------------------------------------------------------------
    # Perform the fit
    # -------------------------------------------------------------------------
    fitresult = model_dist.fit(y_sample, params=model_params, x=x_sample,
                               nan_policy='omit', max_nfev=10000)

    params_odict = {}
    params_odict['function'] = str(func.__name__).lower()
    params_odict['parameters'] = fitresult.params.valuesdict()
    params_odict['fit_statistics'] = {}
    params_odict['fit_statistics']['chisqr'] = fitresult.chisqr
    params_odict['fit_statistics']['max_nfev'] = fitresult.nfev

    # -------------------------------------------------------------------------

    return params_odict


# ====================================================================================

def fit_cdf_model_minimizer(x_sample, y_sample, dist, params_est="undefined"):

    """
    Fits a CDF model to simulated probability data

    :returns:   A dictionary of fitted parameters.
                The structure of the output:

                fitted_model = dict(
                    function=str(),
                    parameters=OrderedDict(),
                    fit_statistics={}
                )

    """

    # ------------------------------------------------------------------------------
    # Locate x-values where the y-values are changing
    x_sample = np.asarray(x_sample)
    y_sample = np.asarray(y_sample)
    change_ndx = np.where(y_sample[:-1] != y_sample[1:])[0]
    change_ndx = list(change_ndx)

    if not (change_ndx[-1] == len(y_sample)):
        change_ndx.insert(len(change_ndx), change_ndx[-1] + 1)

    if change_ndx[0] != 0:
        change_ndx.insert(0, change_ndx[0] - 1)

    xs = x_sample[change_ndx]

    # ------------------------------------------------------------------------------
    # SETUP THE DISTRIBUTION FOR FITTING

    if dist.lower() in ["normal", "normal_cdf"]:
        func_name = "normal_cdf"
        est_mean = np.mean(xs)
        est_sdev = np.std(xs)
        params_est = lmfit.Parameters()
        params_est.add('mean', value=est_mean, min=min(xs), max=est_mean * 2)
        params_est.add('stddev', value=est_sdev, min=0, max=est_mean * 0.9)
        func2min = res_norm_cdf

    elif dist.lower() in ["rayleigh", "rayleigh_cdf"]:
        func_name = "rayleigh_cdf"
        est_loc = xs[0]
        est_scale = np.std(xs)
        params_est = lmfit.Parameters()
        params_est.add('loc', value=est_loc, min=None, max=None)
        params_est.add('scale', value=est_scale, min=None, max=None)
        func2min = res_rayleigh_cdf

    elif dist.lower() in ["lognormal", "lognormal_cdf"]:
        func_name = "lognormal_cdf"
        est_median = np.mean(xs)
        est_beta = np.std(xs)
        est_loc = 0
        if params_est == "undefined":
            params_est = lmfit.Parameters()
            params_est.add(
                'median', value=est_median, min=min(xs), max=est_median * 2)
            params_est.add(
                'beta', value=est_beta, min=sys.float_info.min, max=est_beta * 2)
            params_est.add(
                'loc', value=est_loc, vary=False)
        func2min = res_lognorm_cdf

    else:
        raise ValueError(f"The distribution {dist} is not supported.")

    # ------------------------------------------------------------------------------
    # PARAMETER ESTIMATION

    minimizer_result = lmfit.minimize(
        func2min,
        params_est,
        args=(x_sample, y_sample),
        method='leastsq',
        nan_policy='omit',
        max_nfev=10000
    )
    print(f"Distribution: {func_name}")
    minimizer_result.params.pretty_print()

    # -------------------------------------------------------------------------
    # BUILD THE OUTPUT DICTIONARY

    fitted_model_params = OrderedDict()
    fitted_model_params = minimizer_result.params.valuesdict()

    fitted_model_goodness = {}
    fitted_model_goodness['chisqr'] = minimizer_result.chisqr
    fitted_model_goodness['max_nfev'] = minimizer_result.nfev

    fitted_model = dict(
        function=str(func_name),
        parameters=fitted_model_params,
        fit_statistics=fitted_model_goodness
    )
    # -------------------------------------------------------------------------
    return fitted_model

# ====================================================================================


def fit_prob_exceed_model(
        xdata,
        ydata_2d,
        system_limit_states,
        output_path,
        config_obj,
        distribution='lognormal_cdf',
        CROSSOVER_THRESHOLD=0.005,
        CROSSOVER_CORRECTION=True):
    """
    Fit a Lognormal CDF model to simulated probability exceedance data

    :param xdata: input values for hazard intensity (numpy array)
    :param ydata_2d:   probability of exceedance (2D numpy array)
                        its shape is (num_damage_states x num_hazards_points)
    :param system_limit_states: discrete damage states (list of strings)
    :param output_path: directory path for writing output (string)
    :returns:  fitted exceedance model parameters (PANDAS dataframe)
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # INITIAL SETUP
    # Assumption: the limit_states *INCLUDES* a "DS0 - No Damage" state

    fitted_params_dict = {i: {} for i in range(1, len(system_limit_states))}
    xdata = [float(x) for x in xdata]

    borderA = "=" * 75
    borderB = '-' * 75
    msg_fitresults_init = \
        f"\n\n{borderA}"\
        f"\n{Fore.BLUE}Fitting system FRAGILITY data:{Fore.RESET}"
    rootLogger.info(msg_fitresults_init)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Conduct fitting for given distribution

    for dx in range(1, len(system_limit_states)):

        x_sample = xdata
        y_sample = ydata_2d[dx]
        ds_level = system_limit_states[dx]

        print(borderB)
        print(f"{Fore.YELLOW}Estimated Parameters for: {ds_level}{Fore.RESET}\n")
        params_odict = fit_cdf_model_minimizer(x_sample, y_sample, dist=distribution)
        fitted_params_dict[dx] = params_odict
        # display_dict(params_odict)

    print(f"{borderA}\n")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Check for crossover, and correct as needed

    if CROSSOVER_CORRECTION and (len(system_limit_states[1:]) >= 2):
        # fitted_params_dict = correct_crossover(
        correct_crossover(
            system_limit_states,
            xdata,
            ydata_2d,
            fitted_params_dict,
            CROSSOVER_THRESHOLD)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Calculated set of fitted models are saved as a JSON file

    fitted_params_json = json.dumps(
        {"system_fragility_model": fitted_params_dict}, default=str, indent=4)

    msg_fitresults_final = \
        f"\n\n{color.YELLOW}{color.BOLD}Set of Fitted Models:{color.END}\n\n"\
        f"{fitted_params_json}\n"\
        f"\n{borderB}\n"
    rootLogger.info(msg_fitresults_final)

    json_file = Path(output_path, 'system_model_fragility.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(
            {"system_fragility_model": fitted_params_dict},
            f, ensure_ascii=False, indent=4)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    config_data = dict(
        model_name=config_obj.MODEL_NAME,
        x_param=config_obj.HAZARD_INTENSITY_MEASURE_PARAM,
        x_unit=config_obj.HAZARD_INTENSITY_MEASURE_UNIT,
        scenario_metrics=config_obj.FOCAL_HAZARD_SCENARIOS,
        scneario_names=config_obj.FOCAL_HAZARD_SCENARIO_NAMES
    )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot the simulation data

    plot_data_model(xdata,
                    ydata_2d,
                    system_limit_states,
                    fitted_params_dict,
                    output_path,
                    file_name='fig_sys_pe_DATA.png',
                    PLOT_DATA=True,
                    PLOT_MODEL=False,
                    PLOT_EVENTS=False,
                    **config_data)

    plot_data_model(xdata,
                    ydata_2d,
                    system_limit_states,
                    fitted_params_dict,
                    output_path,
                    file_name='fig_sys_pe_MODEL.png',
                    PLOT_DATA=False,
                    PLOT_MODEL=True,
                    PLOT_EVENTS=False,
                    **config_data)

    plot_data_model(xdata,
                    ydata_2d,
                    system_limit_states,
                    fitted_params_dict,
                    output_path,
                    file_name='fig_sys_pe_MODEL_with_scenarios.png',
                    PLOT_DATA=False,
                    PLOT_MODEL=True,
                    PLOT_EVENTS=True,
                    **config_data)

    return fitted_params_dict

# ====================================================================================


def get_distribution_func(function_name):
    if function_name.lower() in ["normal", "normal_cdf"]:
        return normal_cdf
    elif function_name.lower() in ["rayleigh", "rayleigh_cdf"]:
        return rayleigh_cdf
    elif function_name.lower() in ["lognormal", "lognormal_cdf"]:
        return lognormal_cdf
    else:
        raise ValueError(f"The distribution {function_name} is not supported.")


def plot_data_model(x_vals,
                    y_vals,
                    SYS_DS,
                    model_params,
                    out_path,
                    file_name='fig_sys_pe.png',
                    PLOT_DATA=True,
                    PLOT_MODEL=True,
                    PLOT_EVENTS=False,
                    **conf_data):

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if sum([PLOT_DATA, PLOT_MODEL, PLOT_EVENTS]) == 0:
        raise AttributeError

    model_name = conf_data.get('model_name')
    x_param = conf_data.get('x_param')
    x_unit = conf_data.get('x_unit')
    scenario_metrics = conf_data.get('scenario_metrics')
    scneario_names = conf_data.get('scneario_names')

    plt.style.use('seaborn-darkgrid')
    mpl.rc('grid', linewidth=0.7)
    mpl.rc('font', family='sans-serif')

    colours = spl.ColourPalettes()
    COLR_DS = colours.FiveLevels[-1 * len(SYS_DS):]

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # [Plot 1 of 3] The Data Points

    if PLOT_DATA:
        spl.add_legend_subtitle("DATA")
        for i in range(1, len(SYS_DS)):
            ax.plot(x_vals, y_vals[i],
                    label=SYS_DS[i], clip_on=False, color=COLR_DS[i],
                    linestyle='', alpha=0.6, marker=markers[i - 1],
                    markersize=3, markeredgewidth=1, markeredgecolor=None,
                    zorder=10)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # [Plot 2 of 3] The Fitted Model

    if PLOT_MODEL:
        spl.add_legend_subtitle("FITTED MODEL")
        xmax = max(x_vals)
        xformodel = np.linspace(0, xmax, 101, endpoint=True)
        dmg_mdl_arr = np.zeros((len(SYS_DS), len(xformodel)))

        for dx in range(1, len(SYS_DS)):

            function_name = model_params[dx]['function']
            params = model_params[dx]['parameters']
            distribution = get_distribution_func(function_name)
            dmg_mdl_arr[dx] = distribution(xformodel, **params)

            ax.plot(xformodel, dmg_mdl_arr[dx],
                    label=SYS_DS[dx], clip_on=False, color=COLR_DS[dx],
                    alpha=0.65, linestyle='-', linewidth=1.6,
                    zorder=9)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # [Plot 3 of 3] The Scenario Events

    if PLOT_EVENTS:
        spl.add_legend_subtitle("EVENTS")
        for i, haz in enumerate(scenario_metrics):
            event_num = str(i + 1)
            event_intensity_str = "{:.3f}".format(float(haz))
            event_color = colours.BrewerSpectral[i]
            try:
                event_label = event_num + ". " + \
                    scneario_names[i] + " : " + \
                    event_intensity_str
            except ValueError:
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

    outfig = Path(out_path, file_name)
    figtitle = 'System Fragility: ' + model_name

    x_lab = x_param + ' (' + x_unit + ')'
    y_lab = 'P($D_s$ > $d_s$)'

    y_tick_pos = np.linspace(0.0, 1.0, num=6, endpoint=True)
    y_tick_val = ['{:.1f}'.format(i) for i in y_tick_pos]
    x_tick_pos = np.linspace(0.0, max(x_vals), num=6, endpoint=True)
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

    plt.savefig(outfig,
                format='jpg', dpi=300, bbox_inches='tight')
    plt.close(fig)

# ====================================================================================


def correct_crossover(
        SYS_DS,
        xdata,
        ydata_2d,
        fitted_params_set,
        CROSSOVER_THRESHOLD=0.005):

    """
    Corrects crossovers between sets of algorithms representing damage states.
    This function works only for lognormal cdf's.
    """

    msg_check_crossover = "Checking for crossover [ THRESHOLD = {} ]".format(
        str(CROSSOVER_THRESHOLD))
    rootLogger.info(Fore.GREEN + msg_check_crossover + Fore.RESET)

    params_pe = lmfit.Parameters()

    for dx in range(1, len(SYS_DS)):

        ############################################################################
        x_sample = xdata
        y_sample = ydata_2d[dx]

        mu_hi = fitted_params_set[dx]['parameters']['median']
        sd_hi = fitted_params_set[dx]['parameters']['beta']
        loc_hi = fitted_params_set[dx]['parameters']['loc']

        function_name = fitted_params_set[dx]['function']
        distribution = get_distribution_func(function_name)

        params_hi = fitted_params_set[dx]['parameters']
        y_model_hi = distribution(x_sample, **params_hi)
        MAX = 2 * params_hi['median']

        params_pe.add('median', value=params_hi['median'], min=0, max=MAX)
        params_pe.add('beta', value=params_hi['beta'], min=0, max=MAX)
        params_pe.add('loc', value=0.0, vary=False)

        # fitted_params_set[dx] = lmfit.minimize(
        #     res_lognorm_cdf, params_pe, args=(x_sample, y_sample),
        #     # method='leastsq', nan_policy='omit'
        # )

        fitted_params_set[dx] = fit_cdf_model_minimizer(
            x_sample, y_sample, dist=function_name, params_est=params_pe
        )

        ############################################################################
        if dx >= 2:
            params_lo = fitted_params_set[dx - 1]['parameters']
            mu_lo = fitted_params_set[dx - 1]['parameters']['median']
            sd_lo = fitted_params_set[dx - 1]['parameters']['beta']
            loc_lo = fitted_params_set[dx - 1]['parameters']['loc']
            y_model_lo = distribution(x_sample, **params_lo)

            if abs(min(y_model_lo - y_model_hi)) > CROSSOVER_THRESHOLD:
                msg_overlap_found = \
                    f"{Fore.MAGENTA}"\
                    "There is overlap for curve pair : "\
                    f"{SYS_DS[dx - 1]} - {SYS_DS[dx]}"\
                    f"{Fore.RESET}"
                rootLogger.info(msg_overlap_found)

                # Test if higher curve is co-incident with, or
                # precedes lower curve
                if (mu_hi <= mu_lo) or (loc_hi < loc_lo):
                    rootLogger.info(" *** Mean of higher curve too low: "
                                    "resampling")
                    rootLogger.info("median %s %s", str(mu_hi), str(mu_lo))
                    params_pe.add('median', value=mu_hi, min=mu_lo)
                    # fitted_params_set[dx] = lmfit.minimize(
                    #     res_lognorm_cdf, params_pe, args=(x_sample, y_sample))
                    fitted_params_set[dx] = fit_cdf_model_minimizer(
                        x_sample, y_sample, dist=function_name, params_est=params_pe
                    )
                    (mu_hi, sd_hi, loc_hi) = (
                        fitted_params_set[dx]['parameters']['median'],
                        fitted_params_set[dx]['parameters']['beta'],
                        fitted_params_set[dx]['parameters']['loc'])

                # Thresholds for testing top or bottom crossover
                delta_top = sd_lo - (mu_hi - mu_lo) / 1.0
                delta_btm = sd_lo + (mu_hi - mu_lo) / 1.0

                # Test for top crossover: resample if crossover detected
                if (sd_hi < sd_lo) and (sd_hi <= delta_top):
                    rootLogger.info("*** Attempting to correct upper crossover")
                    params_pe.add('beta', value=sd_hi, min=delta_top)
                    fitted_params_set[dx] = lmfit.minimize(
                        res_lognorm_cdf, params_pe, args=(x_sample, y_sample))

                # Test for bottom crossover: resample if crossover detected
                elif sd_hi >= delta_btm:
                    rootLogger.info("*** Attempting to correct lower crossover")
                    params_pe.add('beta', value=sd_hi, max=delta_btm)
                    fitted_params_set[dx] = lmfit.minimize(
                        res_lognorm_cdf, params_pe, args=(x_sample, y_sample))

            else:
                msg_overlap_none = \
                    f"{Fore.GREEN}"\
                    "There is NO overlap for curve pair: "\
                    f"{SYS_DS[dx - 1]} - {SYS_DS[dx]}"\
                    f"{Fore.RESET}\n"
                rootLogger.info(msg_overlap_none)

    return fitted_params_set

# ====================================================================================
