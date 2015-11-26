from __future__ import print_function

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
import prettyplotlib as ppl
import seaborn as sns

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import lmfit
import pandas as pd

import sys
import os
# import getopt

import siraplot as spl
from siraclasses import Scenario, Facility, PowerStation

import brewer2mpl
from colorama import Fore, Style, init
init()
# stream = AnsiToWin32(sys.stderr).stream

# -----------------------------------------------------------------------------
# READ in SETUP data
# -----------------------------------------------------------------------------

if __name__ == "__main__":

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

# -----------------------------------------------------------------------------
# READ in raw output files from previous system analysis
# -----------------------------------------------------------------------------

economic_loss_array\
    = np.load(os.path.join(RAW_OUTPUT_DIR, 'economic_loss_array.npy'))

calculated_output_array\
    = np.load(os.path.join(RAW_OUTPUT_DIR, 'calculated_output_array.npy'))

output_array_given_recovery\
    = np.load(os.path.join(RAW_OUTPUT_DIR, 'output_array_given_recovery.npy'))

exp_damage_ratio\
    = np.load(os.path.join(RAW_OUTPUT_DIR, 'exp_damage_ratio.npy'))

sys_frag\
    = np.load(os.path.join(RAW_OUTPUT_DIR, 'sys_frag.npy'))

if fc.system_class == 'PowerStation':
    pe_sys\
        = np.load(os.path.join(RAW_OUTPUT_DIR, 'pe_sys_econloss.npy'))
elif fc.system_class == 'Substation':
    pe_sys\
        = np.load(os.path.join(RAW_OUTPUT_DIR, 'pe_sys_cpfailrate.npy'))

required_time\
    = np.load(os.path.join(RAW_OUTPUT_DIR, 'required_time.npy'))

# -----------------------------------------------------------------------------
# Plot customisation
# -----------------------------------------------------------------------------


def ci_dict_to_df(ci):
    convp = lambda x: ('%.2f' % (x[0]*100.0))+'%'
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


def customise_plot(ax, x_lab=None, y_lab=None, figtitle=None, figfile=None,
                   x_scale=None, y_scale=None, x_lim=[], y_lim=[],
                   x_ticks=None, y_ticks=None, x_grid=False, y_grid=False):

    from datetime import datetime
    if figfile is None:
        figfile = 'fig_' +\
            datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S') +\
            '.png'

    grid_colr = '#B6B6B6'
    spine_colr = 'black'
    gridline_wid = 0.2

    # spines_to_keep = ['bottom', 'left', 'top', 'right']
    spines_to_keep = ['bottom']
    for spine in spines_to_keep:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.7)
        ax.spines[spine].set_color(spine_colr)

    spines_to_remove = ['left', 'top', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

    if x_grid:
        ax.xaxis.grid(True, which="both", linestyle='-',
                      linewidth=gridline_wid, color=grid_colr)
    if y_grid:
        ax.yaxis.grid(True, which="major", linestyle='-',
                      linewidth=gridline_wid, color=grid_colr)

    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.set_axisbelow(True)

    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)

    if x_lim:
        ax.set_xlim(x_lim)    # empty list is equivalent to False
    if y_lim:
        ax.set_ylim(y_lim)

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'

    ax.set_xscale('linear') if x_scale is None else ax.set_xscale('log')
    ax.set_yscale('linear') if y_scale is None else ax.set_yscale('log')

    ax.tick_params(
        axis='x',           # changes apply to the x-axis
        which='major',      # ticks affected: major, minor, or both
        bottom='on',        # ticks along the bottom edge are off
        top='on',           # ticks along the top edge are off
        labelbottom='on',   # labels along the bottom edge are off
        color=grid_colr,
        width=gridline_wid,
        length=6)

    ax.tick_params(
        axis='y',           # changes apply to the x-axis
        which='major',      # ticks affected: major, minor, or both
        left='on',          # ticks along the bottom edge are off
        right='on',         # ticks along the top edge are off
        color=grid_colr,
        width=gridline_wid,
        length=0)

    ax.set_title(figtitle, loc='center', y=1.04)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)

    # Shrink current axis by 15%
    box = ax.get_position()
    ax.set_position([box.x0 * 0.9, box.y0 * 1.1,
                     box.width * 0.85, box.height * 0.98])

    ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0),
              frameon=0, prop={'size': 8})

    plt.savefig(figfile, format='png', dpi=300)
    # bbox_inches='tight'
    # plt.close(fig)

# -----------------------------------------------------------------------------
# Damage States and Boundaries
# -----------------------------------------------------------------------------

# sys_dmg_states = ['DS0 None', 'DS1 Slight', 'DS2 Moderate', 'DS3 Extensive',
#                   'DS4 Complete']
# ds_bounds = [0.01, 0.15, 0.4, 0.8, 1.0]

# -----------------------------------------------------------------------------
# For plots: using the  brewer2 color maps by Cynthia Brewer
# -----------------------------------------------------------------------------

clrs = brewer2mpl.get_map('RdYlGn', 'Diverging', 11).mpl_colors
set2 = brewer2mpl.get_map('Set2', 'qualitative', 5).mpl_colors

markers = ['o', '^', 's', 'D', 'x', '+']

###############################################################################
# Restoration over time, given damage state

sys_fn = pd.DataFrame(index=RESTORATION_TIME_RANGE, columns=[SYS_DS])
sys_fn.fillna(1)
sys_fn.index.name = "Time in " + sc.time_unit

for ds in range(len(SYS_DS)):
    fn_tmp = np.zeros((sc.num_hazard_pts, sc.num_time_steps))
    ids = {}     # index of damage states within the samples
    for p in range(sc.num_hazard_pts):
        ids[p] = np.where(sys_frag[:, p] == ds)[0]
        m = np.mean(output_array_given_recovery[ids[p], p, :], axis=0)
        fn_tmp[p] = m / fc.nominal_production
    sys_fn[SYS_DS[ds]] = stats.nanmean(fn_tmp, axis=0)

# sys_fn = sys_fn.drop('DS0 None', axis=1)
sys_fn.to_csv(os.path.join(sc.output_path, 'system_restoration_profile.csv'),
              sep=',')

###############################################################################
# PROBABILITY of EXCEEDENCE
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


def lognorm_cdf(x, shape, loc, scale):
    return stats.lognorm.cdf(x, shape, loc=loc, scale=scale)


def res_lognorm_cdf(params, x, data, eps=None):
    shape = params['logstd'].value
    scale = params['median'].value
    loc = params['loc'].value
    model = stats.lognorm.cdf(x, shape, loc=loc, scale=scale)
    if eps is None:
        return (model - data)
    return (model - data)/eps

fontP = FontProperties()
fontP.set_size('small')

fig = plt.figure(figsize=(9, 5), facecolor='white')
ax = fig.add_subplot(111, axisbg='white')

for i in range(1, len(SYS_DS)):
    ppl.plot(ax,
             sc.hazard_intensity_vals,
             pe_sys[i],
             label=SYS_DS[i], clip_on=False,
             color=spl.COLR_DS[i], linestyle='--', alpha=0.3,
             marker=markers[i-1], markersize=4,
             markeredgecolor=spl.COLR_DS[i])


# DataFrame for storing the calculated System Damage Algorithms for
# exceedance probabilities.
indx = pd.Index(SYS_DS[1:], name='Damage States')
sys_dmg_model = pd.DataFrame(index=indx,
                             columns=['Fragility Median',
                                      'Fragility LogStd',
                                      'Fragility Loc'])

if FIT_PE_DATA:

    # ----- Initial fit -----
    sys_dmg_ci = [{} for _ in xrange(len(SYS_DS))]
    sys_dmg_fit = [[] for _ in xrange(len(SYS_DS))]
    for dx in range(1, len(SYS_DS)):
        x_sample = sc.hazard_intensity_vals
        y_sample = pe_sys[dx]

        p0m = np.mean(y_sample)
        p0s = np.std(y_sample)

        # Fit the dist:
        params_pe = lmfit.Parameters()
        params_pe.add('median', value=p0m) #, min=0, max=10)
        params_pe.add('logstd', value=p0s)
        params_pe.add('loc', value=0.0, vary=False)

        sys_dmg_fit[dx] = lmfit.minimize(res_lognorm_cdf, params_pe,
                                         args=(x_sample, y_sample))

        sys_dmg_model.ix[SYS_DS[dx]]\
            = (sys_dmg_fit[dx].params['median'].value,
               sys_dmg_fit[dx].params['logstd'].value,
               sys_dmg_fit[dx].params['loc'].value)

    print("\n\n"+Fore.YELLOW+"Fitting system fragility data ..."+Fore.RESET)
    print("-"*80)
    print("INITIAL System Fragilities:\n\n", sys_dmg_model, '\n')

    # ----- Check for crossover and resample as needed -----
    for dx in range(1, len(SYS_DS)):
        x_sample = sc.hazard_intensity_vals
        y_sample = pe_sys[dx]

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

        #######################################################################
        if dx >= 2:
            mu_lo, sd_lo, loc_lo =\
                sys_dmg_model.ix[SYS_DS[dx-1]].values
            y_model_lo = stats.lognorm.cdf(x_sample, sd_lo,
                                           loc=loc_lo, scale=mu_lo)

            if sum(y_model_lo-y_model_hi < 0):
                print(Fore.MAGENTA+"There is overlap for curve pair   : " +
                      SYS_DS[dx-1] + '-' + SYS_DS[dx] +
                      Fore.RESET)

                # Test if higher curve is co-incident with,
                # or precedes lower curve
                if (mu_hi <= mu_lo) or (loc_hi <= loc_lo):
                    print("   *** Mean of higher curve too low: resampling")
                    params_pe.add('median', value=mu_hi, min=mu_lo)
                    sys_dmg_fit[dx] = lmfit.minimize(res_lognorm_cdf,
                                                     params_pe,
                                                     args=(x_sample, y_sample))
                    (mu_hi, sd_hi, loc_hi) = (sys_dmg_fit[dx].params['median'].value,
                                              sys_dmg_fit[dx].params['logstd'].value,
                                              sys_dmg_fit[dx].params['loc'].value)

                # Thresholds for testing top or bottom crossover
                delta_top = (3.0*sd_lo - (mu_hi - mu_lo))/3
                delta_btm = (3.0*sd_lo + (mu_hi - mu_lo))/3

                # Test for top crossover: resample if crossover detected
                if (sd_hi < sd_lo) and (sd_hi <= delta_top):
                    print("   *** Attempting to correct upper crossover")
                    params_pe.add('logstd', value=sd_hi, min=delta_top)
                    sys_dmg_fit[dx] = lmfit.minimize(res_lognorm_cdf,
                                                     params_pe,
                                                     args=(x_sample, y_sample))

                # Test for bottom crossover: resample if crossover detected
                # elif (sd_hi >= sd_lo) and sd_hi >= delta_btm:
                elif sd_hi >= delta_btm:
                    print("   *** Attempting to correct lower crossover")
                    params_pe.add('logstd', value=sd_hi, max=delta_btm)
                    sys_dmg_fit[dx] = lmfit.minimize(res_lognorm_cdf,
                                                     params_pe,
                                                     args=(x_sample, y_sample))

            else:
                print(Fore.GREEN +
                      "There is NO overlap for curve pair: " +
                      SYS_DS[dx-1] + '-' + SYS_DS[dx] +
                      Fore.RESET)

        #######################################################################

        sys_dmg_model.ix[SYS_DS[dx]] = \
            sys_dmg_fit[dx].params['median'].value, \
            sys_dmg_fit[dx].params['logstd'].value, \
            sys_dmg_fit[dx].params['loc'].value

        # sys_dmg_ci[dx] = lmfit.conf_interval(sys_dmg_fit[dx], \
        #                                 sigmas=[0.674,0.950,0.997])

    print("\nFINAL System Fragilities: \n")
    print(sys_dmg_model)

    for dx in range(1, len(SYS_DS)):
        print("\n\nFragility model statistics for damage state: %s"
              % SYS_DS[dx])
        print("Goodness-of-Fit chi-square test statistic: %f"
              % sys_dmg_fit[dx].chisqr)
        # print("Confidence intervals: ")
        # lmfit.printfuncs.report_ci(sys_dmg_ci[dx])

    # ----- Plot the fitted models -----
    for dx in range(1, len(SYS_DS)):
        shape = sys_dmg_model.loc[SYS_DS[dx], 'Fragility LogStd']
        loc = sys_dmg_model.loc[SYS_DS[dx], 'Fragility Loc']
        scale = sys_dmg_model.loc[SYS_DS[dx], 'Fragility Median']
        ppl.plot(ax, x_sample,
                 stats.lognorm.cdf(x_sample, shape, loc=loc, scale=scale),
                 label=SYS_DS[dx], clip_on=False,
                 color=spl.COLR_DS[dx], alpha=0.65,
                 linestyle='-', linewidth=1.6)

# xbuffer = min(int(len(x_sample)/10), 5) * (x_sample[2]-x_sample[1])
# ax.set_xlim([min(x_sample)-xbuffer, max(x_sample)+xbuffer])
outfig = os.path.join(sc.output_path, 'fig_MODEL_system_pe.png')
ax.margins(0.03, None)
customise_plot(ax,
               figtitle='System Fragility: '+fc.system_class,
               figfile=outfig,
               x_lab='Peak Ground Acceleration (g)',
               y_lab='P($D_s$ > $d_s$ | PGA)',
               x_scale=None,
               y_scale=None,
               x_ticks=None,
               y_ticks=np.linspace(0.0, 1.0, num=11, endpoint=True),
               x_grid=True,
               y_grid=True)
plt.close(fig)

sys_dmg_model.to_csv(os.path.join(sc.output_path, 'System_Fragility_Model.csv'),
                     sep=',')

###############################################################################
#
# NORMAL CURVE FITTING
#
# Parameters in scipy NORMAL distribution:
#
# The location (loc) keyword specifies the mean.
# The scale (scale) keyword specifies the standard deviation.
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
#
# Note on the covariance matrix returned by scipy.optimize.curve_fit:
# The square root of the diagonal values are the 1-sigma uncertainties of
# the fit parameters
# -----------------------------------------------------------------------------


def norm_cdf(x, mu, sd):
    return stats.norm.cdf(x, loc=mu, scale=sd)


def res_norm_cdf(params, x, data, eps=None):
    mu = params['mean'].value
    sd = params['stddev'].value
    model = stats.norm.cdf(x, loc=mu, scale=sd)
    if eps is None:
        return (model - data)
    return (model - data)/eps


def bimodal_norm_cdf(x, m1, s1, w1, m2, s2, w2):
    return w1*norm_cdf(x, m1, s1) + w2*norm_cdf(x, m2, s2)


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
    return (model - data)/eps

###############################################################################


def fit_restoration_data(ax, SYS_DS, RESTORATION_TIME_RANGE, sys_fn):

    indx = pd.Index(SYS_DS[1:], name='Damage States')
    sys_rst_model1 = pd.DataFrame(index=indx,
                                  columns=['RestoreMean1', 'RestoreStdDev1'])

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

        sys_rst_model1.ix[SYS_DS[dx]]\
            = sys_rst_fit[dx].params['mean'].value,\
              sys_rst_fit[dx].params['stddev'].value

    print("\n\n\n" + Fore.YELLOW +
          "Fitting system restoration data ..." +
          Fore.RESET)
    print("-"*80)
    print("INITIAL Restoration Parameters:\n\n", sys_rst_model1, '\n')

    # ----- Check for crossover and resample as needed -----
    for dx in range(1, len(SYS_DS)):
        x_sample = RESTORATION_TIME_RANGE
        y_sample = sys_fn[SYS_DS[dx]]

        m1_hi = sys_rst_fit[dx].params['mean'].value
        s1_hi = sys_rst_fit[dx].params['stddev'].value
        y_model_hi = norm_cdf(x_sample, m1_hi, s1_hi)

        # ---------------------------------------------------------------------
        # Check for crossover...

        if dx >= 2:
            m1_lo, s1_lo = sys_rst_model1.ix[SYS_DS[dx-1]].values
            y_model_lo = norm_cdf(x_sample, m1_lo, s1_lo)

            if sum(y_model_lo-y_model_hi < 0):
                print(Fore.MAGENTA +
                      "There is overlap for curve pair   : " +
                      SYS_DS[dx-1] + '-' + SYS_DS[dx] +
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
                        params.add('mean', value=m1_hi, min=m1_lo*1.01)
                        sys_rst_fit[dx] = lmfit.minimize(res_norm_cdf, params,
                                                         args=(x_sample,
                                                               y_sample)
                                                         )
                        (m1_hi, s1_hi) = (sys_rst_fit[dx].params['mean'].value,
                                          sys_rst_fit[dx].params['stddev'].value)
                        mu_err += 1

                    # Thresholds for testing top or bottom crossover
                    delta_top = (1+k/100.0)*(3.0*s1_lo - (m1_hi - m1_lo))/3
                    delta_btm = (1-k/100.0)*(3.0*s1_lo + (m1_hi - m1_lo))/3

                    # Test for top crossover: resample if crossover detected
                    if (s1_hi < s1_lo) or (s1_hi <= delta_top):
                        if not sdtop_err > 0:
                            print("   *** " +
                                  "Attempting to correct top crossover...")
                        params.add('mean', value=m1_hi*1.01, min=m1_lo*1.01)
                        params.add('stddev', value=s1_hi, min=delta_top)
                        sys_rst_fit[dx] = lmfit.minimize(res_norm_cdf, params,
                                                         args=(x_sample,
                                                               y_sample)
                                                         )
                        (m1_hi, s1_hi) = (sys_rst_fit[dx].params['mean'].value,
                                          sys_rst_fit[dx].params['stddev'].value)
                        sdtop_err += 1

                    # Test for bottom crossover: resample if crossover detected
                    elif (s1_hi >= delta_btm):
                        if not sdbtm_err > 0:
                            print("   *** " +
                                  "Attempting to correct bottom crossover...")
                        params.add('stddev', value=s1_hi, min=delta_btm)
                        sys_rst_fit[dx] = lmfit.minimize(res_norm_cdf, params,
                                                         args=(x_sample,
                                                               y_sample)
                                                         )
                        (m1_hi, s1_hi) = (sys_rst_fit[dx].params['mean'].value,
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
                      SYS_DS[dx-1] + '-' + SYS_DS[dx] +
                      Fore.RESET)

        # ---------------------------------------------------------------------

        # sys_rst_ci[dx], trace = lmfit.conf_interval(sys_rst_fit[dx], \
        #                     sigmas=[0.674,0.950,0.997], trace=True)

        sys_rst_model1.ix[SYS_DS[dx]]\
            = sys_rst_fit[dx].params['mean'].value,\
              sys_rst_fit[dx].params['stddev'].value

    print("\nFINAL Restoration Parameters: \n")
    print(sys_rst_model1)

    for dx in range(1, len(SYS_DS)):
        print("\n\nRestoration model statistics for damage state: %s"
              % SYS_DS[dx])
        print("Goodness-of-Fit chi-square test statistic: %f"
              % sys_rst_fit[dx].chisqr)
        # print("Confidence intervals: ")
        # lmfit.printfuncs.report_ci(sys_rst_ci[dx])

    return sys_rst_model1, sys_rst_ci

###############################################################################
# RESTORATION CURVES for Discrete Damage States

fig = plt.figure(figsize=(9, 5), facecolor='white')
ax = fig.add_subplot(111, axisbg='white')

# --- Plot simulation data points ---
for i in range(1, len(SYS_DS)):
    ppl.plot(ax,
             RESTORATION_TIME_RANGE[1:],
             sys_fn[SYS_DS[i]][1:] * 100,
             label=SYS_DS[i], clip_on=False,
             color=set2[-i], linestyle='', alpha=0.35,
             marker=markers[i-1], markersize=4, markeredgecolor=set2[-i])

if FIT_RESTORATION_DATA:
    sys_rst_model1, sys_rst_ci = fit_restoration_data(ax, SYS_DS,
                                                      RESTORATION_TIME_RANGE,
                                                      sys_fn)

# --- Plot the fitted models ---
for dx in range(1, len(SYS_DS)):
    m1 = sys_rst_model1.ix[SYS_DS[dx]]['RestoreMean1']
    s1 = sys_rst_model1.ix[SYS_DS[dx]]['RestoreStdDev1']
    ppl.plot(ax, RESTORATION_TIME_RANGE[1:],
             norm_cdf(RESTORATION_TIME_RANGE, m1, s1)[1:] * 100,
             label=SYS_DS[dx], clip_on=False, color=set2[-dx],
             linestyle='-', linewidth=1.5, alpha=.85)

# plt.axvline(x=99, ymin=0.0, ymax=1.0,
#             linestyle='-', linewidth=0.2, color='#B6B6B6')
outfig = os.path.join(sc.output_path, 'fig_MODEL_sys_rst_mode1.png')
ax.margins(0.03, None)
customise_plot(ax,
               figtitle='Restoration Curves: '+fc.system_class,
               figfile=outfig,
               x_lab='Time ('+sc.time_unit+')',
               y_lab='Percent Functional',
               x_scale='log',   # <OR> None
               y_scale=None,
               x_ticks=None,
               y_ticks=np.linspace(0, 100, num=6, endpoint=True),
               x_grid=True,
               y_grid=True)
plt.close(fig)

###############################################################################

# sturges = lambda n: int(np.log2(n) + 1)
# sys_fn[DS].hist(bins=sturges(sys_fn[DS].size), normed=True,
#     color='lightseagreen')
# sys_fn[DS].dropna().plot(kind='kde', xlim=(0,100), style='r--')
# plt.show(block=False)

sns.set_style("whitegrid",
              {"legend.frameon": True,
               "grid.linewidth": 0.3,
               "grid.color": ".9"})

indx = pd.Index(SYS_DS[1:], name='Damage States')
sys_rst_model2 = pd.DataFrame(index=indx,
                              columns=['BimodalMean1', 'BimodalStd1',
                                       'BimodalWeight1', 'BimodalMean2',
                                       'BimodalStd2', 'BimodalWeight2'])

sys_mix_fit = [[] for _ in xrange(len(SYS_DS))]
# sys_mix_ci  = [{} for _ in xrange(len(SYS_DS))]

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

    m1 = params_mx['m1'].value
    s1 = params_mx['s1'].value
    w1 = params_mx['w1'].value
    m2 = params_mx['m2'].value
    s2 = params_mx['s2'].value
    w2 = params_mx['w2'].value

    # sys_mix_ci[dx] = lmfit.conf_interval(sys_mix_fit[dx], \
    #                     sigmas=[0.674,0.950,0.997], trace=False)

    sys_rst_model2.ix[DS] = m1, s1, w1, m2, s2, w2

# sys_rst_ci_df = ci_dict_to_df(sys_mix_ci)

print("\n\n"+"="*80)
print("System Restoration Parameters:\n\n", sys_rst_model2, '\n')

fig = plt.figure(figsize=(9, 5), facecolor='white')
ax = fig.add_subplot(111)

A = [ax.lines for x in range(len(SYS_DS))]
B = [ax.lines for x in range(len(SYS_DS))]
C = [ax.lines for x in range(len(SYS_DS))]

for dx in range(1, len(SYS_DS)):
    DS = SYS_DS[dx]
    x_sample = RESTORATION_TIME_RANGE

    print("\n\nBimodal Normal Model | " + \
          "Restoration model statistics for damage state: %s" % DS)
    print("Goodness-of-Fit chi-square test statistic: %f" \
          % sys_mix_fit[dx].chisqr)
    # print("Confidence intervals: ")
    # lmfit.printfuncs.report_ci(sys_mix_ci[dx])

    A[dx], = ax.plot(x_sample,
                     sys_fn[DS].values * 100, label=DS+" Data",
                     marker=markers[dx-1], markersize=4, clip_on=False,
                     linestyle='', color=set2[-dx], alpha=0.4)

    B[dx], = ax.plot(x_sample,
                     norm_cdf(x_sample, sys_rst_model1.ix[DS]['RestoreMean1'],
                              sys_rst_model1.ix[DS]['RestoreStdDev1'])*100,
                     label=DS+" Single-mode", clip_on=False,
                     linestyle='--', linewidth=0.75,
                     color=set2[-dx], alpha=0.7)

    C[dx], = ax.plot(x_sample,
                     bimodal_norm_cdf(x_sample,
                                      *sys_rst_model2.ix[DS].values)*100,
                     label=DS+" Bimodal",
                     linestyle='-', linewidth=1.5,
                     color=set2[-dx], alpha=0.95)

print("\n"+"-"*80)

figtitle = "Restoration Models"
x_lab = "Time ("+sc.time_unit+")"
y_lab = "Percent Functional"

ax.set_title(figtitle, loc='center', y=1.04)
ax.set_xlabel(x_lab, fontsize=10)
ax.set_ylabel(y_lab, fontsize=10)
ax.set_ylim([0, 100])
ax.set_xlim([0, sc.restore_time_upper])
# ax.set_xscale('log')

outfig = os.path.join(sc.output_path, 'fig_MODEL_sys_rst_mode2.png')
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(8)
box = ax.get_position()
ax.set_position([box.x0 * 0.9, box.y0 * 1.1,
                 box.width * 0.85, box.height * 0.98])

A[0], = plt.plot([0], marker='None', linestyle='None', label='dummy_1')
B[0], = plt.plot([0], marker='None', linestyle='None', label='dummy_2')
C[0], = plt.plot([0], marker='None', linestyle='None', label='dummy_3')

ax.legend(A+B+C,
          ["DATA"] + SYS_DS[1:] +
          ["\n"+"Model 1:"+"\nNormal, 1 Component"] + SYS_DS[1:] +
          ["\n"+"Model 2:"+"\nNormal, 2 Components"] + SYS_DS[1:],
          loc='upper left', bbox_to_anchor=(1.01, 1.0),
          frameon=0, ncol=1, numpoints=1, prop={'size': 7})
plt.savefig(outfig, format='png', dpi=300, bbox_inches='tight')
plt.close(fig)

###############################################################################

sys_rst_params = sys_rst_model1.join([sys_rst_model2])
sys_rst_params.to_csv(os.path.join(sc.output_path,
                                   'System_Restoration_Model.csv'), sep=',')

###############################################################################
