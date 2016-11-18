from __future__ import print_function
from sifraclasses import *

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import lmfit
import pandas as pd

import sys
import os

import sifraplot as spl

import brewer2mpl
from colorama import Fore, Style, init

init()


# stream = AnsiToWin32(sys.stderr).stream

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def ci_dict_to_df(ci):
    convp = lambda x: ('%.2f' % (x[0] * 100.0)) + '%'
    conv = lambda x: x[1]
    ci_header = []
    ci_values = []
    title_set = False
    for name, row in ci.items():
        if not title_set:
            ciheader = [i for i in map(convp, row)]
            title_set = True
        ci_values.append([i for i in map(conv, row)])
    ci_df = pd.DataFrame(ci_values, index=ci.keys(), columns=ci_header)
    ci_df = ci_df.sort()
    return ci_df


# ----------------------------------------------------------------------------
# For plots: using the  brewer2 color maps by Cynthia Brewer
# ----------------------------------------------------------------------------

clrs = brewer2mpl.get_map('RdYlGn', 'Diverging', 11).mpl_colors
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


def lognorm_cdf(x, shape, loc, scale):
    return stats.lognorm.cdf(x, shape, loc=loc, scale=scale)


def res_lognorm_cdf(params, x, data, eps=None):
    shape = params['logstd'].value
    scale = params['median'].value
    loc = params['loc'].value
    model = stats.lognorm.cdf(x, shape, loc=loc, scale=scale)
    if eps is None:
        return (model - data)
    return (model - data) / eps


# ============================================================================

def fit_prob_exceed_model(hazard_input_vals, pb_exceed, SYS_DS, out_path):
    """
    Fit a Lognormal CDF model to simulated probability exceedance data

    :param hazard_input_vals: input values for hazard intensity (numpy array)
    :param pb_exceed: probability of exceedance (2D numpy array)
    :param SYS_DS: discrete damage states (list)
    :param out_path: directory path for writing output (string)
    :returns:  fitted exceedance model parameters (PANDAS dataframe)
    """
    # DataFrame for storing the calculated System Damage Algorithms for
    # exceedance probabilities.
    indx = pd.Index(SYS_DS[1:], name='Damage States')
    sys_dmg_model = pd.DataFrame(index=indx,
                                 columns=['Median',
                                          'LogStdDev',
                                          'Location',
                                          'Chi-Sqr'])

    # ----- Initial fit -----
    sys_dmg_ci = [{} for _ in xrange(len(SYS_DS))]
    sys_dmg_fit = [[] for _ in xrange(len(SYS_DS))]
    for dx in range(1, len(SYS_DS)):
        x_sample = hazard_input_vals
        y_sample = pb_exceed[dx]

        p0m = np.mean(y_sample)
        p0s = np.std(y_sample)

        # Fit the dist:
        params_pe = lmfit.Parameters()
        params_pe.add('median', value=p0m)  # , min=0, max=10)
        params_pe.add('logstd', value=p0s)
        params_pe.add('loc', value=0.0, vary=False)

        sys_dmg_fit[dx] = lmfit.minimize(res_lognorm_cdf, params_pe,
                                         args=(x_sample, y_sample))

        sys_dmg_model.ix[SYS_DS[dx]] \
            = (sys_dmg_fit[dx].params['median'].value,
               sys_dmg_fit[dx].params['logstd'].value,
               sys_dmg_fit[dx].params['loc'].value,
               sys_dmg_fit[dx].chisqr)

    print("\n" + "-" * 79)
    print(Fore.YELLOW +
          "Fitting system FRAGILITY data: Lognormal CDF" +
          Fore.RESET)
    print("-" * 79)
    print("INITIAL System Fragilities:\n\n", sys_dmg_model, '\n')

    # ----- Check for crossover and resample as needed -----
    for dx in range(1, len(SYS_DS)):
        x_sample = hazard_input_vals
        y_sample = pb_exceed[dx]

        mu_hi = sys_dmg_fit[dx].params['median'].value
        sd_hi = sys_dmg_fit[dx].params['logstd'].value
        loc_hi = sys_dmg_fit[dx].params['loc'].value

        y_model_hi = stats.lognorm.cdf(x_sample, sd_hi,
                                       loc=loc_hi, scale=mu_hi)

        params_pe.add('median', value=mu_hi, min=0, max=10)
        params_pe.add('logstd', value=sd_hi)
        params_pe.add('loc', value=0.0, vary=False)
        sys_dmg_fit[dx] = lmfit.minimize(res_lognorm_cdf, params_pe,
                                         args=(x_sample, y_sample))

        ######################################################################
        if dx >= 2:
            mu_lo, sd_lo, loc_lo, chi = \
                sys_dmg_model.ix[SYS_DS[dx - 1]].values
            y_model_lo = stats.lognorm.cdf(x_sample, sd_lo,
                                           loc=loc_lo, scale=mu_lo)

            if sum(y_model_lo - y_model_hi < 0):
                print(Fore.MAGENTA + "There is overlap for curve pair   : " +
                      SYS_DS[dx - 1] + '-' + SYS_DS[dx] +
                      Fore.RESET)

                # Test if higher curve is co-incident with,
                # or precedes lower curve
                if (mu_hi <= mu_lo) or (loc_hi <= loc_lo):
                    print("   *** Mean of higher curve too low: resampling")
                    params_pe.add('median', value=mu_hi, min=mu_lo)
                    sys_dmg_fit[dx] = lmfit.minimize(
                        res_lognorm_cdf, params_pe, args=(x_sample, y_sample))

                    (mu_hi, sd_hi, loc_hi) = \
                        (sys_dmg_fit[dx].params['median'].value,
                         sys_dmg_fit[dx].params['logstd'].value,
                         sys_dmg_fit[dx].params['loc'].value)

                # Thresholds for testing top or bottom crossover
                delta_top = (3.0 * sd_lo - (mu_hi - mu_lo)) / 3
                delta_btm = (3.0 * sd_lo + (mu_hi - mu_lo)) / 3

                # Test for top crossover: resample if crossover detected
                if (sd_hi < sd_lo) and (sd_hi <= delta_top):
                    print("   *** Attempting to correct upper crossover")
                    params_pe.add('logstd', value=sd_hi, min=delta_top)
                    sys_dmg_fit[dx] = lmfit.minimize(
                        res_lognorm_cdf, params_pe, args=(x_sample, y_sample))

                # Test for bottom crossover: resample if crossover detected
                # elif (sd_hi >= sd_lo) and sd_hi >= delta_btm:
                elif sd_hi >= delta_btm:
                    print("   *** Attempting to correct lower crossover")
                    params_pe.add('logstd', value=sd_hi, max=delta_btm)
                    sys_dmg_fit[dx] = lmfit.minimize(
                        res_lognorm_cdf, params_pe, args=(x_sample, y_sample))

            else:
                print(Fore.GREEN +
                      "There is NO overlap for curve pair: " +
                      SYS_DS[dx - 1] + '-' + SYS_DS[dx] +
                      Fore.RESET)

        ######################################################################

        sys_dmg_model.ix[SYS_DS[dx]] = \
            sys_dmg_fit[dx].params['median'].value, \
            sys_dmg_fit[dx].params['logstd'].value, \
            sys_dmg_fit[dx].params['loc'].value, \
            sys_dmg_fit[dx].chisqr

        # sys_dmg_ci[dx] = lmfit.conf_interval(sys_dmg_fit[dx], \
        #                                 sigmas=[0.674,0.950,0.997])

    print("\nFINAL System Fragilities: \n")
    print(sys_dmg_model)

    # for dx in range(1, len(SYS_DS)):
    #     print("\n\nFragility model statistics for damage state: %s"
    #           % SYS_DS[dx])
    #     print("Goodness-of-Fit chi-square test statistic: %f"
    #           % sys_dmg_fit[dx].chisqr)
    #     print("Confidence intervals: ")
    #     lmfit.printfuncs.report_ci(sys_dmg_ci[dx])

    # ----- Write fitted model params to file -----
    sys_dmg_model.to_csv(os.path.join(out_path,
                                      'system_model_fragility.csv'), sep=',')

    # ----- Plot the simulation data -----
    fontP = FontProperties()
    fontP.set_size('small')

    fig = plt.figure(figsize=(9, 4.5), facecolor='white')
    ax = fig.add_subplot(111, axisbg='white')

    spl.add_legend_subtitle("Data")

    for i in range(1, len(SYS_DS)):
        ax.plot(hazard_input_vals,
                pb_exceed[i],
                label=SYS_DS[i], clip_on=False,
                color=spl.COLR_DS[i], linestyle='', alpha=0.3,
                marker=markers[i - 1], markersize=4,
                markeredgecolor=spl.COLR_DS[i])

    # ----- Plot the fitted models -----
    dmg_mdl_arr = np.zeros((len(SYS_DS), len(hazard_input_vals)))
    # plt.plot([0], marker='None', linestyle='None',
    #          label="\nFitted Model: LogNormal")

    spl.add_legend_subtitle("\nFitted Model: LogNormal CDF")

    for dx in range(1, len(SYS_DS)):
        shape = sys_dmg_model.loc[SYS_DS[dx], 'LogStdDev']
        loc = sys_dmg_model.loc[SYS_DS[dx], 'Location']
        scale = sys_dmg_model.loc[SYS_DS[dx], 'Median']
        dmg_mdl_arr[dx] = stats.lognorm.cdf(
            x_sample, shape, loc=loc, scale=scale)
        ax.plot(hazard_input_vals,
                dmg_mdl_arr[dx],
                label=SYS_DS[dx], clip_on=False,
                color=spl.COLR_DS[dx], alpha=0.65,
                linestyle='-', linewidth=1.6)

    # xbuffer = min(int(len(x_sample)/10), 5) * (x_sample[2]-x_sample[1])
    # ax.set_xlim([min(x_sample)-xbuffer, max(x_sample)+xbuffer])
    # ax.margins(0.03, None)
    outfig = os.path.join(out_path, 'fig_MODEL_sys_pb_exceed.png')
    spl.format_fig(ax,
                   figtitle='System Fragility: ' + fc.system_class,
                   x_lab='Peak Ground Acceleration (g)',
                   y_lab='P($D_s$ > $d_s$ | PGA)',
                   x_scale=None,
                   y_scale=None,
                   x_tick_val=None,
                   y_tick_val=np.linspace(0.0, 1.0, num=11, endpoint=True),
                   x_grid=False,
                   y_grid=True,
                   add_legend=True)

    # ----- Finish plotting -----
    plt.savefig(outfig, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    return sys_dmg_model


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
    sys_rst_ci = [{} for _ in xrange(len(SYS_DS))]
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

        sys_rst_mdl_mode1.ix[SYS_DS[dx]] \
            = sys_rst_fit[dx].params['mean'].value, \
              sys_rst_fit[dx].params['stddev'].value, \
              sys_rst_fit[dx].chisqr

    print("\n\n" + "-" * 79)
    print(Fore.YELLOW +
          "Fitting system RESTORATION data: Unimodal Normal CDF" +
          Fore.RESET)
    print("-" * 79)
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
            m1_lo, s1_lo, r1_chi = sys_rst_mdl_mode1.ix[SYS_DS[dx - 1]].values
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

        sys_rst_mdl_mode1.ix[SYS_DS[dx]] \
            = sys_rst_fit[dx].params['mean'].value, \
              sys_rst_fit[dx].params['stddev'].value, \
              sys_rst_fit[dx].chisqr

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
        m1 = sys_rst_mdl_mode1.ix[SYS_DS[dx]]['Mean']
        s1 = sys_rst_mdl_mode1.ix[SYS_DS[dx]]['StdDev']
        ax.plot(RESTORATION_TIME_RANGE[1:],
                norm_cdf(RESTORATION_TIME_RANGE, m1, s1)[1:] * 100,
                label=SYS_DS[dx], clip_on=False, color=spl.COLR_DS[dx],
                linestyle='-', linewidth=1.5, alpha=0.65)

    x_pwr = int(np.ceil(np.log10(max(RESTORATION_TIME_RANGE))))
    x_tiks = [10 ** t for t in range(0, x_pwr + 1)]

    outfig = os.path.join(out_path, 'fig_MODEL_sys_rst_mode1.png')
    ax.margins(0.03, None)
    spl.format_fig(ax,
                   figtitle='Restoration Curves: ' + fc.system_class,
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

        sys_rst_mdl_mode2.ix[DS] = m1, s1, w1, m2, s2, w2, \
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
                x_sample, *sys_rst_mdl_mode2.ix[DS].values[:-1])[1:] * 100,
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

def approximate_generic_sys_restoration(sc, fc, sys_frag,
                                        output_array_given_recovery):
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
            m = np.mean(output_array_given_recovery[ids[p], p, :], axis=0)
            fn_tmp[p] = m / fc.nominal_production
        sys_fn[SYS_DS[ds]] = stats.nanmean(fn_tmp, axis=0)

    # sys_fn = sys_fn.drop('DS0 None', axis=1)
    sys_fn.to_csv(os.path.join(
        sc.output_path, 'system_restoration_profile.csv'), sep=',')

    return sys_fn


# ============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------------
    # READ in SETUP data
    # The first argument is the full path to the config file
    SETUPFILE = sys.argv[1]
    discard = {}
    config = {}

    # Read in config file and define Scenario & Facility objects
    exec (open(SETUPFILE).read(), discard, config)
    FacilityObj = eval(config["SYSTEM_CLASS"])
    sc = Scenario(SETUPFILE)
    fc = FacilityObj(SETUPFILE)

    # Define input files, output location, scenario inputs
    INPUT_PATH = os.path.join(os.getcwd(), sc.input_dir_name)
    SYS_CONFIG_FILE = os.path.join(INPUT_PATH, fc.sys_config_file_name)
    RAW_OUTPUT_DIR = sc.raw_output_dir
    RESTORATION_TIME_RANGE = sc.restoration_time_range
    SYS_DS = fc.sys_dmg_states

    # Test switches
    FIT_PE_DATA = sc.fit_pe_data
    FIT_RESTORATION_DATA = sc.fit_restoration_data

    # ------------------------------------------------------------------------
    # READ in raw output files from prior analysis of system fragility

    economic_loss_array = \
        np.load(os.path.join(
            RAW_OUTPUT_DIR, 'economic_loss_array.npy'))

    calculated_output_array = \
        np.load(os.path.join(
            RAW_OUTPUT_DIR, 'calculated_output_array.npy'))

    output_array_given_recovery = \
        np.load(os.path.join(
            RAW_OUTPUT_DIR, 'output_array_given_recovery.npy'))

    exp_damage_ratio = \
        np.load(os.path.join(
            RAW_OUTPUT_DIR, 'exp_damage_ratio.npy'))

    sys_frag = \
        np.load(os.path.join(
            RAW_OUTPUT_DIR, 'sys_frag.npy'))

    required_time = \
        np.load(os.path.join(RAW_OUTPUT_DIR, 'required_time.npy'))

    if fc.system_class == 'PowerStation':
        pe_sys = \
            np.load(os.path.join(RAW_OUTPUT_DIR, 'pe_sys_econloss.npy'))
    elif fc.system_class == 'Substation':
        pe_sys = \
            np.load(os.path.join(RAW_OUTPUT_DIR, 'pe_sys_cpfailrate.npy'))

    # ------------------------------------------------------------------------
    # Calculate & Plot Fitted Models

    sys_fn = approximate_generic_sys_restoration(sc, fc, sys_frag,
                                                 output_array_given_recovery)

    if FIT_PE_DATA:
        sys_dmg_model = fit_prob_exceed_model(
            sc.hazard_intensity_vals, pe_sys, SYS_DS, sc.output_path)

    if FIT_RESTORATION_DATA:
        sys_rst_mdl_mode1 = fit_restoration_data(
            RESTORATION_TIME_RANGE, sys_fn, SYS_DS, sc.output_path)
        # sys_rst_mdl_mode2 = fit_restoration_data_multimode(
        #     RESTORATION_TIME_RANGE, sys_fn, SYS_DS, scn.output_path)
        print("\n" + "-" * 79)

# ============================================================================
