import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Union

import lmfit
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from colorama import Fore, init
from scipy import stats
from scipy.special import erf  # noqa:E0611

import sira.tools.siraplot as spl

init()
rootLogger = logging.getLogger(__name__)
pd.options.display.float_format = "{:,.4f}".format

# -----------------------------------------------------------------------------
# For plots: using the  brewer2 color maps by Cynthia Brewer
# -----------------------------------------------------------------------------

set2 = sns.color_palette("Set2", 5)
markers = ["o", "^", "s", "D", "x", "+"]


class color:
    """From: https://stackoverflow.com/a/17303428"""

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


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
# Note on the covariance matrix returned by scipy.optimise.curve_fit:
# The square root of the diagonal values are the 1-sigma uncertainties of
# the fit parameters.
# -----------------------------------------------------------------------------


def lognormal_cdf(x, median, beta, loc=0):
    x = np.asarray(x)
    with np.errstate(divide="ignore"):
        logn_cdf = 0.5 + 0.5 * erf((np.log(x - loc) - np.log(median)) / (np.sqrt(2) * beta))
    return logn_cdf


def normal_cdf(x, mean, stddev):
    return 0.5 * (1 + erf((x - mean) / (stddev * np.sqrt(2))))


def rayleigh_cdf(x, loc, scale):
    return stats.rayleigh.cdf(x, loc=loc, scale=scale)


# ====================================================================================


def display_dict(pdict, width=4, level=0, init_space=0):
    """Neatly prints a dict of params"""
    ip = init_space
    for key, value in pdict.items():
        if isinstance(value, float):
            val_fmt = f"{value:<.4f}"
        else:
            val_fmt = f"{str(value):<12}"
        if not (isinstance(value, dict) or isinstance(value, OrderedDict)) and (level < 1):
            print(" " * init_space + "[" + str(key) + "]")
            print(f"{' ' * (init_space + width)}{val_fmt}")
        if isinstance(value, dict):
            print(" " * (init_space + width * level) + "[" + str(key) + "]")
            display_dict(value, level=level + 1, init_space=ip)
        elif level > 0:
            print(f"{' ' * (init_space + width * level)}{str(key):<10} = {val_fmt}")


def fit_cdf_model(
    x_sample,
    y_sample,
    dist,
    params_est: Union[lmfit.Parameters, dict, None] = None,
    tag=None,
):
    """
    Fits given array-like data using `lmfit.Model` method

    :returns:   A dictionary of fitted parameters.
                The structure of the output:

                fitted_model = OrderedDict(
                    function=str(),
                    parameters=OrderedDict(),
                    fit_statistics={}
                )
    """

    # Locate x-values where the y-values are changing
    x_sample = np.array(x_sample)
    y_sample = np.array(y_sample)

    if y_sample.size == 0:
        rootLogger.error(f"{Fore.RED}y-values array is empty.\n{Fore.RESET}")
        raise ValueError()
    if np.all(y_sample == 0):
        rootLogger.error(f"{Fore.RED}y-values array is all zeros.\n{Fore.RESET}")
        raise ValueError()
    if len(np.unique(y_sample)) == 1:
        rootLogger.error(
            f"{Fore.RED}y-values all values in array are same. Nothing to fit.\n{Fore.RESET}"
        )
        raise ValueError()

    change_ndx = np.where(y_sample[:-1] != y_sample[1:])[0]
    change_ndx = list(change_ndx)

    if not (change_ndx[-1] == len(y_sample)):
        change_ndx.insert(len(change_ndx), change_ndx[-1] + 1)

    if change_ndx[0] != 0:
        change_ndx.insert(0, change_ndx[0] - 1)

    try:
        xs = x_sample[change_ndx]

        # -------------------------------------------------------------------------
        # NORMAL CDF -- set up model and params
        # -------------------------------------------------------------------------
        if dist.lower() in ["normal", "gaussian", "normal_cdf"]:
            func = normal_cdf
            model_dist = lmfit.Model(func)
            model_params = model_dist.make_params()
            if params_est is None or params_est == "undefined":
                model_params.add("mean", value=np.mean(xs), min=min(xs), max=float(np.mean(xs)) * 2)
                model_params.add("stddev", value=np.std(xs), min=0, max=float(np.mean(xs)) * 0.9)
            else:
                param = "mean"
                model_params.add(
                    param,
                    value=params_est[param].value,
                    min=params_est[param].min,
                    max=params_est[param].max,
                )
                param = "stddev"
                model_params.add(
                    param,
                    value=params_est[param].value,
                    min=params_est[param].min,
                    max=params_est[param].max,
                )

        # -------------------------------------------------------------------------
        # LOGNORMAL CDF -- set up model and params
        # -------------------------------------------------------------------------
        elif dist.lower() in ["lognormal", "lognormal_cdf"]:
            func = lognormal_cdf
            model_dist = lmfit.Model(func)
            init_med = float(np.mean(xs))
            init_lsd = abs(float(np.std(xs)))
            model_params = model_dist.make_params()
            if params_est is None or params_est == "undefined":
                model_params.add("median", value=init_med, min=min(xs), max=init_med * 2)
                model_params.add("beta", value=init_lsd, min=sys.float_info.min, max=init_med * 2)
                model_params.add("loc", value=0, vary=False)
            else:
                param = "median"
                model_params.add(
                    param,
                    value=params_est[param].value,
                    min=params_est[param].min,
                    max=params_est[param].max,
                )
                param = "beta"
                model_params.add(
                    param,
                    value=params_est[param].value,
                    min=params_est[param].min,
                    max=params_est[param].max,
                )
                model_params.add("loc", value=0, vary=False)

        # -------------------------------------------------------------------------
        # RAYLEIGH CDF -- set up model and params
        # -------------------------------------------------------------------------
        elif dist.lower() in ["rayleigh", "rayleigh_cdf"]:
            func = rayleigh_cdf
            model_dist = lmfit.Model(func)
            init_loc = xs[0]
            init_scale = np.std(xs)
            model_params = model_dist.make_params()
            model_params.add("loc", value=init_loc)
            model_params.add("scale", value=init_scale)

        else:
            raise ValueError(f"The distribution {dist} is not supported.")

    except Exception as e:
        rootLogger.error(f"{Fore.RED}Error setting up the fit model: {e}\n{Fore.RESET}")
        raise e

    # -------------------------------------------------------------------------
    # Perform the fit
    # -------------------------------------------------------------------------
    fitresult = model_dist.fit(
        y_sample, params=model_params, x=x_sample, nan_policy="omit", max_nfev=10000
    )

    params_odict = OrderedDict()
    params_odict["function"] = str(func.__name__).lower()
    params_odict["parameters"] = fitresult.params.valuesdict()
    params_odict["fit_statistics"] = OrderedDict()
    params_odict["fit_statistics"]["chisqr"] = fitresult.chisqr
    params_odict["fit_statistics"]["max_nfev"] = fitresult.nfev

    # -------------------------------------------------------------------------
    func_name = params_odict["function"]
    if tag is not None:
        fit_data_header = f"{Fore.YELLOW}{tag} | Distribution: {func_name}{Fore.RESET}"
    else:
        fit_data_header = f"{Fore.RESET}Distribution: {func_name}{Fore.RESET}"
    print("\n" + "-" * 88)
    print(fit_data_header)
    print("-" * 88)
    # print(fitresult.fit_report(modelpars=fitresult.params))
    display_dict(params_odict)

    return params_odict


# ====================================================================================


def fit_prob_exceed_model(
    xdata,
    ydata_2d,
    system_limit_states,
    config_data_dict,
    output_path,
    distribution="lognormal_cdf",
    CROSSOVER_THRESHOLD=0.005,
    CROSSOVER_CORRECTION=True,
):
    """
    Fit a Lognormal CDF model to simulated probability exceedance data

    :param xdata: input values for hazard intensity (numpy array)
    :param ydata_2d: probability of exceedance (2D numpy array)
        its shape is (num_damage_states x num_hazards_points)
    :param system_limit_states: discrete damage states (list of strings)
    :param output_path: directory path for writing output (string)
    :returns:  fitted exceedance model parameters (PANDAS dataframe)
    """

    xdata = np.array(xdata)
    ydata_2d = np.array(ydata_2d)

    # Debug prints
    print("\nDebug - Damage States:", system_limit_states)
    print("Debug - ydata_2d shape:", ydata_2d.shape)
    print("Debug - First few values of ydata_2d:\n", ydata_2d[:, :5])  # Show first 5 columns

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # INITIAL SETUP
    # Assumption: the limit_states *INCLUDES* a "DS0 - No Damage" state

    fitted_params_dict = {i: {} for i in range(1, len(system_limit_states))}
    xdata = [float(x) for x in xdata]

    borderA = "=" * 80
    borderB = "-" * 80
    msg_fitresults_init = f"{Fore.BLUE}Fitting system FRAGILITY data...{Fore.RESET}"
    rootLogger.info(msg_fitresults_init)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Conduct fitting for given distribution

    for dx in range(1, len(system_limit_states)):
        x_sample = xdata
        y_sample = ydata_2d[dx]
        ds_level = system_limit_states[dx]

        print(f"\nDebug info - Processing DS{dx}: {ds_level}")
        print(f"Debug info - y_sample mean: {np.mean(y_sample)}")
        print(f"Debug info - y_sample min/max: {np.min(y_sample)}/{np.max(y_sample)}")

        params_odict = fit_cdf_model(
            x_sample, y_sample, dist=distribution, tag=f"Limit State: {ds_level}"
        )
        fitted_params_dict[dx] = params_odict

    print(f"\n{borderA}\n")
    print(params_odict)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Check for crossover, and correct as needed

    if CROSSOVER_CORRECTION and (len(system_limit_states[1:]) >= 2):
        fitted_params_dict = correct_crossover(
            system_limit_states, xdata, ydata_2d, fitted_params_dict, CROSSOVER_THRESHOLD
        )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Calculated set of fitted models are saved as a JSON file

    fitted_params_json = json.dumps(
        {"system_fragility_model": fitted_params_dict}, default=str, indent=4
    )

    msg_fitresults_final = (
        f"\n{borderA}\n"
        f"\n{color.YELLOW}{color.BOLD}Set of Fitted Models:{color.END}\n"
        f"{fitted_params_json}\n"
        f"\n{borderB}\n"
    )
    rootLogger.info(msg_fitresults_final)

    json_file = Path(output_path, "system_model_fragility.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"system_fragility_model": fitted_params_dict}, f, ensure_ascii=False, indent=4)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot the simulation data

    plot_data_model(
        xdata,
        ydata_2d,
        system_limit_states,
        fitted_params_dict,
        output_path,
        file_name="fig_sys_pe_DATA.png",
        PLOT_DATA=True,
        PLOT_MODEL=False,
        PLOT_EVENTS=False,
        **config_data_dict,
    )

    plot_data_model(
        xdata,
        ydata_2d,
        system_limit_states,
        fitted_params_dict,
        output_path,
        file_name="fig_sys_pe_MODEL.png",
        PLOT_DATA=False,
        PLOT_MODEL=True,
        PLOT_EVENTS=False,
        **config_data_dict,
    )

    plot_data_model(
        xdata,
        ydata_2d,
        system_limit_states,
        fitted_params_dict,
        output_path,
        file_name="fig_sys_pe_MODEL_with_scenarios.png",
        PLOT_DATA=False,
        PLOT_MODEL=True,
        PLOT_EVENTS=True,
        **config_data_dict,
    )

    return fitted_params_dict


def get_distribution_func(function_name):
    if function_name.lower() in ["normal", "normal_cdf"]:
        return normal_cdf
    elif function_name.lower() in ["rayleigh", "rayleigh_cdf"]:
        return rayleigh_cdf
    elif function_name.lower() in ["lognormal", "lognormal_cdf"]:
        return lognormal_cdf
    else:
        raise ValueError(f"The distribution {function_name} is not supported.")


def plot_data_model(
    x_vals,
    y_vals,
    SYS_DS,
    model_params,
    out_path,
    file_name="fig_sys_pe.png",
    PLOT_DATA=True,
    PLOT_MODEL=True,
    PLOT_EVENTS=False,
    **conf_data,
):
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if sum([PLOT_DATA, PLOT_MODEL, PLOT_EVENTS]) == 0:
        raise AttributeError

    model_name = conf_data.get("model_name")
    x_param = conf_data.get("x_param")
    x_unit = conf_data.get("x_unit")
    scenario_metrics = conf_data.get("scenario_metrics")
    scneario_names = conf_data.get("scneario_names")

    # `scenario_metrics` is expected to be a list of hazard intensity values
    if scenario_metrics is None or len(scenario_metrics) == 0:
        PLOT_EVENTS = False
        rootLogger.warning("No metrics found for plotting scenario events.")
        rootLogger.warning("Skipping plotting of scenario events.\n")

    try:
        scenario_metrics = [float(h) for h in scenario_metrics]  # type: ignore
    except (TypeError, ValueError) as e:
        PLOT_EVENTS = False
        rootLogger.warning(f"Invalid metrics found for scenario events: {e}")
        rootLogger.warning("Skipping plotting of scenario events.\n")

    mpl.rc("grid", linewidth=0.7)
    mpl.rc("font", family="sans-serif")

    colours = spl.ColourPalettes()
    COLR_DS = colours.FiveLevels[-1 * len(SYS_DS) :]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def calc_xtick_vals(x_max, XTICK_GAP, format_string="{:.1f}"):
        NUM_XTICKS = int(np.round(x_max / XTICK_GAP) + 1)
        xtick_pos = np.linspace(0.0, x_max, num=NUM_XTICKS, endpoint=True)
        xtick_val = [format_string.format(i) for i in xtick_pos]
        return xtick_pos, xtick_val

    x_max = float(max(x_vals))
    if x_max <= 1.0:
        XTICK_GAP = 0.1
    elif x_max <= 3.0:
        XTICK_GAP = 0.2
    elif x_max <= 5.0:
        XTICK_GAP = 0.5
    elif x_max <= 10:
        XTICK_GAP = 1
    elif x_max <= 50:
        XTICK_GAP = 5
    elif x_max <= 100:
        XTICK_GAP = 10
    else:
        XTICK_GAP = int(x_max / 10)
    if x_max <= 1.0:
        sfmt = "{:.2f}"
    else:
        sfmt = "{:.1f}"

    x_unit = str(x_unit)
    if x_max is None or x_max in ["", "nan", "na"]:
        if x_unit.lower() == "g":
            x_max = 1.4
            XTICK_GAP = 0.2
            sfmt = "{:.2f}"
        elif x_param.lower() in ["sa", "spectral acceleration"]:
            x_max = 15
            XTICK_GAP = 1
            sfmt = "{:.0f}"
        elif x_unit.lower() in ["m/s^2", "m/s2"]:
            x_max = 10
            XTICK_GAP = 1
            sfmt = "{:.0f}"
        elif x_param.lower() in ["k", "k-out-of-n"]:
            x_max = 5
            XTICK_GAP = 1
            sfmt = "{:.0f}"

    if x_unit.lower() not in ["g", "m/s^2", "m/s2", "k", "k-out-of-n", "none", "na", "nan"]:
        print(f"X-axis unit found is: {x_unit}.")
        raise ValueError("X-axis unit must be g or m/s^2")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    # --------------------------------------------------------------------------
    # [Plot 1 of 3] The Data Points

    if PLOT_DATA:
        spl.add_legend_subtitle("DATA")
        for i in range(1, len(SYS_DS)):
            ax.plot(
                x_vals,
                y_vals[i],
                label=SYS_DS[i],
                clip_on=False,
                color=COLR_DS[i],
                linestyle="",
                alpha=0.6,
                marker=markers[i - 1],
                markersize=3,
                markeredgewidth=1,
                markeredgecolor=None,
                zorder=10,
            )

    # --------------------------------------------------------------------------
    # [Plot 2 of 3] The Fitted Model

    if PLOT_MODEL:
        spl.add_legend_subtitle("FITTED MODEL")
        xformodel = np.linspace(0, x_max, 121, endpoint=True)
        dmg_mdl_arr = np.zeros((len(SYS_DS), len(xformodel)), dtype=float)

        for dx in range(1, len(SYS_DS)):
            function_name = model_params[dx]["function"]
            params = model_params[dx]["parameters"]
            distribution = get_distribution_func(function_name)
            dmg_mdl_arr[dx] = distribution(xformodel, **params)

            ax.plot(
                xformodel,
                dmg_mdl_arr[dx],
                label=SYS_DS[dx],
                clip_on=False,
                color=COLR_DS[dx],
                alpha=0.65,
                linestyle="-",
                linewidth=1.6,
                zorder=9,
            )

    # --------------------------------------------------------------------------
    # [Plot 3 of 3] The Scenario Events

    if PLOT_EVENTS:
        spl.add_legend_subtitle("EVENTS")
        for i, haz in enumerate(scenario_metrics):  # type: ignore
            event_num = str(i + 1)
            event_intensity_str = "{:.3f}".format(float(haz))
            event_color = colours.GreenArmytage[i]
            try:
                event_label = event_num + ". " + scneario_names[i] + " : " + event_intensity_str
            except ValueError:
                event_label = event_num + " : " + event_intensity_str

            ax.plot(
                float(haz),
                0,
                label=event_label,
                color=event_color,
                marker="",
                markersize=2,
                linestyle="-",
                zorder=11,
            )
            ax.plot(
                float(haz),
                1.04,
                label="",
                clip_on=False,
                color=event_color,
                marker="o",
                fillstyle="none",
                markersize=12,
                linestyle="-",
                markeredgewidth=1.0,
                zorder=11,
            )
            ax.annotate(
                event_num,  # event_intensity_str,
                xy=(float(haz), 0),
                xycoords="data",
                xytext=(float(haz), 1.038),
                textcoords="data",
                ha="center",
                va="center",
                rotation=0,
                size=8,
                fontweight="bold",
                color=event_color,
                annotation_clip=False,
                bbox=dict(boxstyle="round, pad=0.2", fc="yellow", alpha=0.0),
                path_effects=[PathEffects.withStroke(linewidth=2, foreground="w")],
                arrowprops=dict(
                    arrowstyle="-|>, head_length=0.5, head_width=0.3",
                    shrinkA=3.0,
                    shrinkB=0.0,
                    connectionstyle="arc3,rad=0.0",
                    color=event_color,
                    alpha=0.8,
                    linewidth=1.0,
                    linestyle="-",
                    path_effects=[PathEffects.withStroke(linewidth=2.5, foreground="w")],
                ),
                zorder=11,
            )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ax.set_axisbelow("line")

    outfig = Path(out_path, file_name)
    figtitle = f"System Fragility: {model_name}"

    x_tick_pos, x_tick_val = calc_xtick_vals(x_max, XTICK_GAP=XTICK_GAP, format_string=sfmt)
    # x_tick_pos = np.linspace(0.0, max(x_vals), num=6, endpoint=True)
    # x_tick_val = ['{:.2f}'.format(i) for i in x_tick_pos]

    y_tick_pos = np.linspace(0.0, 1.0, num=11, endpoint=True)
    y_tick_val = ["{:.1f}".format(i) for i in y_tick_pos]

    if x_unit is None or x_unit.lower() in ["none", "na"]:
        x_lab = x_param
    else:
        x_lab = f"{x_param} ({x_unit})"
    y_lab = "Probability of Exceedance  P($D_s$ > $d_s$)"

    ax.set_title(figtitle, loc="center", y=1.04, fontweight="bold", size=10)
    ax.set_xlabel(x_lab, size=10, labelpad=10)  # type: ignore
    ax.set_ylabel(y_lab, size=10, labelpad=10)

    ax.set_xlim(0, max(x_tick_pos))
    ax.set_xticks(x_tick_pos)
    ax.set_xticklabels(x_tick_val, size=9)

    ax.set_ylim(0, max(y_tick_pos))
    ax.set_yticks(y_tick_pos)
    ax.set_yticklabels(y_tick_val, size=9)

    # ==============================================================
    # Apply 'bokeh' style to axes
    spl.prettify_axes(ax)
    # ==============================================================
    ax.tick_params(axis="both", pad=7)
    ax.margins(0, 0)

    # Shrink current axis width by 15%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # type: ignore

    # Put a legend to the right of the current axis
    ax.legend(
        title="",
        loc="upper left",
        ncol=1,
        bbox_to_anchor=(1.02, 1.0),
        frameon=0,
        prop={"size": 9},
        alignment="left",
    )

    plt.savefig(outfig, format="jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ====================================================================================


def correct_crossover(SYS_DS, xdata, ydata_2d, fitted_params_set, CROSSOVER_THRESHOLD=0.005):
    """
    Corrects crossovers between sets of algorithms representing damage states.
    This function works only for lognormal cdf's.
    """

    msg_check_crossover = f"\nChecking for crossover [ THRESHOLD = {str(CROSSOVER_THRESHOLD)} ]"
    rootLogger.info(Fore.GREEN + msg_check_crossover + Fore.RESET)

    params_pe = lmfit.Parameters()

    ds_iter = iter(range(2, len(SYS_DS)))
    for dx in ds_iter:
        x_sample = xdata
        y_sample = ydata_2d[dx]

        function_name = fitted_params_set[dx]["function"]
        distribution = get_distribution_func(function_name)

        param_names = list(fitted_params_set[dx]["parameters"].keys())
        param_1 = param_names[0]
        param_2 = param_names[1]

        # --------------------------------------------------------------------------
        params_hi = fitted_params_set[dx]["parameters"]
        y_model_hi = distribution(x_sample, **params_hi)

        mu_hi = fitted_params_set[dx]["parameters"][param_1]
        sd_hi = fitted_params_set[dx]["parameters"][param_2]

        MAX = 2 * params_hi[param_1]

        params_pe.add(param_1, value=params_hi[param_1], min=0, max=MAX)
        params_pe.add(param_2, value=params_hi[param_2], min=0, max=MAX)

        # --------------------------------------------------------------------------
        params_lo = fitted_params_set[dx - 1]["parameters"]
        y_model_lo = distribution(x_sample, **params_lo)

        mu_lo = fitted_params_set[dx - 1]["parameters"][param_1]
        sd_lo = fitted_params_set[dx - 1]["parameters"][param_2]

        if abs(max(y_model_lo - y_model_hi)) > CROSSOVER_THRESHOLD:
            # Test if higher curve is co-incident with, or exceeds lower curve
            # Note: `loc` param for lognorm assumed zero
            if mu_hi <= mu_lo:
                cx_msg_1 = (
                    f"\n {Fore.MAGENTA}*** Mean of higher curve too low: resampling{Fore.RESET}"
                )
                cx_msg_2 = f"{Fore.MAGENTA}{param_1}: {str(mu_hi)} {str(mu_lo)} {Fore.RESET}"
                rootLogger.info(cx_msg_1)
                rootLogger.info(cx_msg_2)
                params_pe.add(param_1, value=mu_hi, min=mu_lo)
                fitted_params_set[dx] = fit_cdf_model(
                    x_sample,
                    y_sample,
                    dist=function_name,
                    params_est=params_pe,
                    tag=f"Limit State: {SYS_DS[dx]} | crossover correction attempt",
                )

                (mu_hi, sd_hi) = (
                    fitted_params_set[dx]["parameters"][param_1],
                    fitted_params_set[dx]["parameters"][param_2],
                )

            # Thresholds for testing top or bottom crossover
            delta_top = sd_lo - (mu_hi - mu_lo)
            delta_btm = sd_lo + (mu_hi - mu_lo)

            # Test for top crossover: resample if crossover detected
            if (sd_hi < sd_lo) and (sd_hi <= delta_top):
                rootLogger.info(
                    "%s*** Attempting to correct upper crossover%s", Fore.MAGENTA, Fore.RESET
                )
                params_pe.add(param_2, value=sd_hi, min=delta_top)
                fitted_params_set[dx] = fit_cdf_model(
                    x_sample,
                    y_sample,
                    dist=function_name,
                    params_est=params_pe,
                    tag=f"Limit State: {SYS_DS[dx]} | crossover correction attempt",
                )

            # Test for bottom crossover: resample if crossover detected
            elif sd_hi >= delta_btm:
                rootLogger.info(
                    "%s*** Attempting to correct lower crossover%s", Fore.MAGENTA, Fore.RESET
                )
                params_pe.add(param_2, value=sd_hi, max=delta_btm)
                fitted_params_set[dx] = fit_cdf_model(
                    x_sample,
                    y_sample,
                    dist=function_name,
                    params_est=params_pe,
                    tag=f"Limit State: {SYS_DS[dx]} | crossover correction attempt",
                )

        # --------------------------------------------------------------------------

    return fitted_params_set


# ====================================================================================
