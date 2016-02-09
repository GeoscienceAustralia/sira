'''
siraplot.py
This module provides easy access to selected colours from the Brewer
palettes, and functions for customising and improving plot aesthetics
'''

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['legend.numpoints'] = 2
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'

from datetime import datetime
import numpy as np
import brewer2mpl
import re

# ----------------------------------------------------------------------------

COLR_DARK2 = brewer2mpl.get_map('Dark2', 'Qualitative', 8).mpl_colors
COLR_SET1 = brewer2mpl.get_map('Set1', 'Qualitative', 9).mpl_colors
COLR_SET2 = brewer2mpl.get_map('Set2', 'Qualitative', 8).mpl_colors
COLR_SET3 = brewer2mpl.get_map('Set3', 'Qualitative', 12).mpl_colors
COLR_RDYLGN = brewer2mpl.get_map('RdYlGn', 'Diverging', 11).mpl_colors
COLR_PAIR = brewer2mpl.get_map('Paired', 'Qualitative', 12).mpl_colors
COLR_SPECTRAL = brewer2mpl.get_map('Spectral', 'Diverging', 11).mpl_colors
COLR_DS = [COLR_PAIR[9], COLR_PAIR[3], COLR_PAIR[1],
           COLR_PAIR[7], COLR_PAIR[5]]
COLR_MIX = COLR_SET1 + COLR_DARK2

# ----------------------------------------------------------------------------


def split_long_label(string, delims, max_chars_per_line=20):
    '''
    Splits long labels into smaller chunks for better print/display outcome
    '''
    delims = [' ', '_']
    pattern = r'\s*(%s)\s*' % ('|'.join((re.escape(d) for d in delims)))
    splt_str = [i for i in re.split(pattern, string) if i and i is not None]

    str_list = []
    lines = []
    for i, val in enumerate(splt_str):
        str_list.append(val)
        if len(''.join(str_list)) >= max_chars_per_line and \
           (i < len(splt_str) - 1):
            str_list.append('\n')
            lines.extend(str_list)
            str_list = []

    if str_list != []:
        lines.extend(str_list)
    lines = ''.join(lines)
    return lines

# ----------------------------------------------------------------------------


def calc_tick_pos(stepsize, ax_vals_list, ax_labels_list,
                  maxnumticks=11, plot_type='line'):
    '''
    Calculates appropriate tick positions based on
    given input parameters
    '''
    stepsize = stepsize
    numticks = int(round((max(ax_vals_list) - min(ax_vals_list)) / stepsize))

    while numticks > maxnumticks:
        stepsize = stepsize * 2.0
        numticks = int(round((max(ax_vals_list) - min(ax_vals_list)) /
                             stepsize))

    skip = int(len(ax_vals_list) / numticks)
    ndx_all = range(1, len(ax_vals_list) + 1, 1)

    if plot_type == 'box':
        tick_pos = ndx_all[0::skip]
        if max(tick_pos) != max(ndx_all):
            numticks += 1
            tick_pos = np.append(tick_pos, max(ndx_all))

        tick_val = np.zeros(len(tick_pos))
        i = 0
        for j in tick_pos:
            tick_val[i] = ax_labels_list[j - 1]
            i += 1

    elif plot_type == 'line':
        tick_pos = ax_vals_list[0::skip]
        if max(tick_pos) != max(ax_vals_list):
            numticks += 1
            tick_pos = np.append(tick_pos, max(ax_vals_list))
        tick_val = tick_pos

    else:
        tick_pos = ax_vals_list
        tick_val = ax_labels_list

    return tick_pos, tick_val

# ----------------------------------------------------------------------------

def add_legend_subtitle(str):
    """
    Places a subtitle over the legend.
    Useful for plots with multiple groups of legends.
    :param str: sub-title for legend
    """
    plt.plot([0], marker='None', linestyle='None',
             label=str)

# ----------------------------------------------------------------------------

def forceAspect(ax,aspect=1):
    """
    Forces the aspect ratio to be equal
    Copy of Yann's answer to the SO question:
    http://stackoverflow.com/questions/7965743/\
        how-can-i-set-the-aspect-ratio-in-matplotlib

    :param ax:
    :param aspect:
    """
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

# ----------------------------------------------------------------------------

def format_fig(axis, x_lab=None, y_lab=None, figtitle=None,
               x_scale=None, y_scale=None,
               x_tick_pos=None, y_tick_pos=None,
               x_tick_val=None, y_tick_val=None,
               x_lim=[], y_lim=[],
               x_grid=False, y_grid=False,
               x_margin=None, y_margin=None,
               add_legend=False, legend_title=None,
               aspectratio=0):
    '''
    Customises plots to a clean appearance and color choices from the
    'brewer' palettes
    '''

    # figfile=None; save_file=False
    # if figfile is None:
    #     figfile = 'fig_' +\
    #               datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S') +\
    #               '.png'

    grid_colr = '#B6B6B6'   # '#E6E6E6'
    spine_colr = 'black'    # '#555555'

    spines_to_keep = ['bottom', 'left']
    for spine in spines_to_keep:
        axis.spines[spine].set_visible(True)
        axis.spines[spine].set_linewidth(0.7)
        axis.spines[spine].set_color(spine_colr)

    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        axis.spines[spine].set_visible(False)

    axis.xaxis.grid(False)
    axis.yaxis.grid(False)

    if x_grid:
        axis.xaxis.grid(True, which="major", linestyle='-',
                        linewidth=0.5, color=grid_colr)
    if y_grid:
        axis.yaxis.grid(True, which="major", linestyle='-',
                        linewidth=0.5, color=grid_colr)

    axis.xaxis.labelpad = 12
    axis.yaxis.labelpad = 12
    axis.set_axisbelow(True)

    if x_scale is not None:
        if x_scale.lower()=='linear':
            axis.set_xscale('linear')
        elif x_scale.lower()=='log':
            axis.set_xscale('log')

    if y_scale is not None:
        if y_scale.lower()=='linear':
            axis.set_yscale('linear')
        elif y_scale.lower()=='log':
            axis.set_yscale('log')

    axis.tick_params(
        axis='x',           # changes apply to the x-axis
        which='both',       # ticks affected: major, minor, or both
        bottom='on',        # ticks along the bottom edge are on
        top='off',          # ticks along the top edge are off
        labelbottom='on',   # labels along the bottom edge are off
        color=spine_colr,
        direction='out',
        labelsize=7,
        pad=5,
        width=0.5,
        length=4)

    axis.tick_params(
        axis='y',
        which='major',
        left='on',
        right='off',
        labelleft='on',
        labelright='off',
        color=spine_colr,
        direction='out',
        labelsize=7,
        pad=5,
        width=0.5,
        length=4)

    if x_tick_pos is not None:
        axis.set_xticks(x_tick_pos)
    if y_tick_pos is not None:
        axis.set_yticks(y_tick_pos)

    if x_tick_val is not None:
        axis.set_xticklabels(x_tick_val)
    if y_tick_val is not None:
        axis.set_yticklabels(y_tick_val)

    axis.margins(x_margin, y_margin)

    axis.set_title(figtitle, loc='center', y=1.04)
    axis.set_xlabel(x_lab)
    axis.set_ylabel(y_lab)

    axis.title.set_fontsize(10)
    # for item in [axis.xaxis.label, axis.yaxis.label]: item.set_fontsize(10)

    # Shrink current axis width by 15%
    box = axis.get_position()
    axis.set_position([box.x0,
                       box.y0,
                       box.width * 0.85,
                       box.height])

    # Put a legend to the right of the current axis
    if add_legend is True:
        axis.legend(title=legend_title,
                    loc='upper left', ncol=1, bbox_to_anchor=(1.02, 1.0),
                    frameon=0, prop={'size': 7})

    if aspectratio > 0:
        forceAspect(axis, aspect=aspectratio)

    if len(x_lim) == 2:
        axis.set_xlim(x_lim)
    if len(y_lim) == 2:
        axis.set_ylim(y_lim)

    # if save_file is True:
    #     plt.savefig(figfile, format='png', bbox_inches='tight', dpi=250)
    # plt.close(fig)

# ----------------------------------------------------------------------------
