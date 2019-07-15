'''
siraplot.py
This module provides easy access to selected colours from the Brewer
palettes, and functions for customising and improving plot aesthetics
'''

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
import brewer2mpl
import re

mpl.rcParams['legend.numpoints'] = 1
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['grid.linewidth'] = 0.5
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = \
#     ['Droid Serif'] + mpl.rcParams['font.serif']
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams.update()
# mpl.font_manager._rebuild()

class ColourPalettes(object):

    def __init__(self):
        # ---------------------------------------------------------------------
        # List of 26 colours with maximal contrast as proposed by
        # Paul Green-Armytage in:
        # P. Green-Armytage (2010). "A Colour Alphabet and the Limits of
        # Colour Coding". Colour: Design & Creativity (5) (2010): 10, 1-23

        self.__GreenArmytage = [
            '#F0A3FF', '#0075DC', '#993F00', '#4C005C', '#191919',
            '#005C31', '#2BCE48', '#FFCC99', '#808080', '#94FFB5',
            '#8F7C00', '#9DCC00', '#C20088', '#003380', '#FFA405',
            '#FFA8BB', '#426600', '#FF0010', '#5EF1F2', '#00998F',
            '#E0FF66', '#740AFF', '#990000', '#FFFF80', '#FFFF00',
            '#FF5005'
            ]

        # ---------------------------------------------------------------------
        # Sasha Trubetskoy's list of 20 simple distinct colours
        # https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
        #
        # Red       #e6194b	(230, 25, 75)	(0, 100, 66, 0)
        # Green     #3cb44b	(60, 180, 75)	(75, 0, 100, 0)
        # Yellow    #ffe119	(255, 225, 25)	(0, 25, 95, 0)
        # Blue      #0082c8	(0, 130, 200)	(100, 35, 0, 0)
        # Orange    #f58231	(245, 130, 48)	(0, 60, 92, 0)
        # Purple    #911eb4	(145, 30, 180)	(35, 70, 0, 0)
        # Cyan      #46f0f0	(70, 240, 240)	(70, 0, 0, 0)
        # Magenta   #f032e6	(240, 50, 230)	(0, 100, 0, 0)
        # Lime      #d2f53c	(210, 245, 60)	(35, 0, 100, 0)
        # Pink      #fabebe	(250, 190, 190)	(0, 30, 15, 0)
        # Teal      #008080	(0, 128, 128)	(100, 0, 0, 50)
        # Lavender  #e6beff	(230, 190, 255)	(10, 25, 0, 0)
        # Brown     #aa6e28	(170, 110, 40)	(0, 35, 75, 33)
        # Beige     #fffac8	(255, 250, 200)	(5, 10, 30, 0)
        # Maroon    #800000	(128, 0, 0)	(0, 100, 100, 50)
        # Mint      #aaffc3	(170, 255, 195)	(33, 0, 23, 0)
        # Olive     #808000	(128, 128, 0)	(0, 0, 100, 50)
        # Coral     #ffd8b1	(255, 215, 180)	(0, 15, 30, 0)
        # Navy      #000080	(0, 0, 128)	    (100, 100, 0, 50)
        # Grey      #808080	(128, 128, 128)	(0, 0, 0, 50)
        # White     #FFFFFF	(255, 255, 255)	(0, 0, 0, 0)
        # Black     #000000	(0, 0, 0)	    (0, 0, 0, 100)

        self.__Trubetskoy = [
            "#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231",
            "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#fabebe",
            "#008080", "#e6beff", "#aa6e28", "#fffac8", "#800000",
            "#aaffc3", "#808000", "#ffd8b1", "#000080", "#808080"]

        # ---------------------------------------------------------------------
        # 269 maximally different colours
        # https://stackoverflow.com/a/33295456

        self.__Tartarize269 = [
            "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941",
            "#006FA6", "#A30059", "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC",
            "#B79762", "#004D43", "#8FB0FF", "#997D87", "#5A0007", "#809693",
            "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
            "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9",
            "#B903AA", "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299",
            "#300018", "#0AA6D8", "#013349", "#00846F", "#372101", "#FFB500",
            "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
            "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68",
            "#7A87A1", "#788D66", "#885578", "#FAD09F", "#FF8A9A", "#D157A0",
            "#BEC459", "#456648", "#0086ED", "#886F4C", "#34362D", "#B4A8BD",
            "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
            "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757",
            "#C8A1A1", "#1E6E00", "#7900D7", "#A77500", "#6367A9", "#A05837",
            "#6B002C", "#772600", "#D790FF", "#9B9700", "#549E79", "#FFF69F",
            "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
            "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804",
            "#324E72", "#6A3A4C", "#83AB58", "#001C1E", "#D1F7CE", "#004B28",
            "#C8D0F6", "#A3A489", "#806C66", "#222800", "#BF5650", "#E83000",
            "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
            "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94",
            "#7ED379", "#012C58", "#7A7BFF", "#D68E01", "#353339", "#78AFA1",
            "#FEB2C6", "#75797C", "#837393", "#943A4D", "#B5F4FF", "#D2DCD5",
            "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176",
            "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75",
            "#8D8546", "#9695C5", "#E773CE", "#D86A78", "#3E89BE", "#CA834E",
            "#518A87", "#5B113C", "#55813B", "#E704C4", "#00005F", "#A97399",
            "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01",
            "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63",
            "#4C6001", "#9C6966", "#64547B", "#97979E", "#006A66", "#391406",
            "#F4D749", "#0045D2", "#006C31", "#DDB6D0", "#7C6571", "#9FB2A4",
            "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C",
            "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527",
            "#8BB400", "#797868", "#C6005A", "#3B000A", "#C86240", "#29607C",
            "#402334", "#7D5A44", "#CCB87C", "#B88183", "#AA5199", "#B5D6C3",
            "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
            "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177",
            "#BCB1E5", "#76912F", "#003109", "#0060CD", "#D20096", "#895563",
            "#29201D", "#5B3213", "#A76F42", "#89412E", "#1A3A2A", "#494B5A",
            "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F",
            "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23",
            "#47675D", "#3A3F00", "#061203", "#DFFB71", "#868E7E", "#98D058",
            "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66", "#2F5D9B", "#6C5E46",
            "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25",
            "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"]

        # ---------------------------------------------------------------------
        # Brewer qualitative colour palettes

        self.__BrewerSet1 = \
            brewer2mpl.get_map('Set1', 'Qualitative', 9).mpl_colors
        self.__BrewerSet2 = \
            brewer2mpl.get_map('Set2', 'Qualitative', 8).mpl_colors
        self.__BrewerSet3 = \
            brewer2mpl.get_map('Set3', 'Qualitative', 12).mpl_colors
        self.__BrewerDark = \
            brewer2mpl.get_map('Dark2', 'Qualitative', 8).mpl_colors
        self.__BrewerPaired = \
            brewer2mpl.get_map('Paired', 'Qualitative', 12).mpl_colors
        self.__BrewerSpectral = \
            brewer2mpl.get_map('Spectral', 'Diverging', 11).mpl_colors

        self.__FiveLevels = [self.__BrewerPaired[9],
                             self.__BrewerPaired[3],
                             self.__BrewerPaired[1],
                             self.__BrewerPaired[7],
                             self.__BrewerPaired[5]]
        # ---------------------------------------------------------------------

    @property
    def GreenArmytage(self):
        return self.__GreenArmytage

    @property
    def Trubetskoy(self):
        return self.__Trubetskoy

    @property
    def BrewerSet1(self):
        return self.__BrewerSet1

    @property
    def BrewerSet2(self):
        return self.__BrewerSet2

    @property
    def BrewerSet3(self):
        return self.__BrewerSet3

    @property
    def BrewerDark(self):
        return self.__BrewerDark

    @property
    def BrewerPaired(self):
        return self.__BrewerPaired

    @property
    def BrewerSpectral(self):
        return self.__BrewerSpectral

    @property
    def FiveLevels(self):
        return self.__FiveLevels

    @property
    def Tartarize269(self):
        return self.__Tartarize269

    def get(self, attr):
        return getattr(self, attr)


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


# def calc_tick_pos(stepsize, ax_vals_list, ax_labels_list,
#                   maxnumticks=11, plot_type='line'):
#     '''
#     Calculates appropriate tick positions based on
#     given input parameters
#     '''
#     stepsize = stepsize
#     numticks = int(round((max(ax_vals_list) - min(ax_vals_list)) / stepsize))
#
#     while numticks > maxnumticks:
#         stepsize = stepsize * 2.0
#         numticks = int(round((max(ax_vals_list) - min(ax_vals_list)) /
#                              stepsize))
#
#     skip = int(len(ax_vals_list) / numticks)
#     ndx_all = range(1, len(ax_vals_list) + 1, 1)
#
#     if plot_type == 'box':
#         tick_pos = ndx_all[0::skip]
#         if max(tick_pos) != max(ndx_all):
#             numticks += 1
#             tick_pos = np.append(tick_pos, max(ndx_all))
#
#         tick_val = np.zeros(len(tick_pos))
#         i = 0
#         for j in tick_pos:
#             tick_val[i] = ax_labels_list[j - 1]
#             i += 1
#
#     elif plot_type == 'line':
#         tick_pos = ax_vals_list[0::skip]
#         if max(tick_pos) != max(ax_vals_list):
#             numticks += 1
#             tick_pos = np.append(tick_pos, max(ax_vals_list))
#         tick_val = tick_pos
#
#     else:
#         tick_pos = ax_vals_list
#         tick_val = ax_labels_list
#
#     return tick_pos, tick_val

# ----------------------------------------------------------------------------

def add_legend_subtitle(str):
    """
    Places a subtitle over the legend.
    Useful for plots with multiple groups of legends.
    :param str: sub-title for legend
    """
    label = "\n" + ' '.join(['$\\bf{'+i+'}$' for i in str.split(' ')])
    plt.plot([0], marker='None', linestyle='None',
             label=label)

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

def format_fig(axis, figtitle=None, x_lab=None, y_lab=None,
               x_scale=None, y_scale=None,
               x_tick_pos=None, y_tick_pos=None,
               x_tick_val=None, y_tick_val=None,
               x_lim=[], y_lim=[],
               x_grid=False, y_grid=False,
               x_margin=None, y_margin=None,
               add_legend=False, legend_title=None,
               aspectratio=0):
    """
    Customises plots to a clean consistent appearance
    """

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
        bottom=True,        # ticks along the bottom edge are on
        top=False,          # ticks along the top edge are off
        labelbottom=True,   # labels along the bottom edge are off
        color=spine_colr,
        direction='out',
        labelsize=7,
        pad=0,
        width=0.5,
        length=4)

    axis.tick_params(
        axis='y',
        which='major',
        left=True,
        right=False,
        labelleft=True,
        labelright=False,
        color=spine_colr,
        direction='out',
        labelsize=7,
        pad=0,
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

    if len(x_lim) == 2:
        axis.set_xlim(x_lim)
    if len(y_lim) == 2:
        axis.set_ylim(y_lim)

    axis.set_title(figtitle, loc='center', y=1.04, fontweight='bold', size=11)
    axis.set_xlabel(x_lab, size=10)
    axis.set_ylabel(y_lab, size=10)

    # axis.title.set_fontsize(11)
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

# ----------------------------------------------------------------------------
