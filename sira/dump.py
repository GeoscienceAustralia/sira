__author__ = 'sudipta'
import getopt
from colorama import Fore

def main(argv):
    setup_file = ''
    msg = ''
    try:
        opts, args = getopt.getopt(argv, "s:", ["setup="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-s", "--setup"):
            setup_file = arg

    if setup_file == '':
        setup_file = "setup.conf"
        msg = "(using default conf filename)"

    print("\n" + "Setup file: " + Fore.YELLOW + setup_file + Fore.RESET + msg)
    return setup_file

# -----------------------------------------------------------------------------
# Simulation Parameters
# -----------------------------------------------------------------------------

dmg_states = sorted([str(d) for d in FRAGILITIES.index.levels[1]])

# max_recoverytimes_dict = {}
# for x in cp_types_in_system:
#     max_recoverytimes_dict[x] =\
#         FRAGILITIES.ix[x, dmg_states[len(dmg_states) - 1]]['recovery_mean']

restoration_time_range, time_step =\
    np.linspace(0, RESTORE_TIME_UPPER, num=RESTORE_TIME_UPPER+1,
                endpoint=USE_ENDPOINT, retstep=True)

restoration_chkpoints, restoration_pct_steps =\
    np.linspace(0.0, 1.0, RESTORE_PCT_CHKPOINTS, retstep=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

hazard_data_points = int(round((PGA_MAX - PGA_MIN) / float(PGA_STEP) + 1))

hazard_intensity_vals = np.linspace(PGA_MIN, PGA_MAX,
                         num=hazard_data_points)

num_hazard_pts = len(hazard_intensity_vals)
num_time_steps = len(restoration_time_range)

# -----------------------------------------------------------------------------
# Define the system as a network, with components as nodes
# -----------------------------------------------------------------------------

nodes_all = sorted(COMP_DF.index)
num_elements = len(nodes_all)

#                    ------
# Network setup with igraph (for analysis)
#                    ------
G = igraph.Graph(directed=True)
nodes = COMP_DF.index.tolist()

G.add_vertices(len(nodes))
G.vs["name"] = nodes
G.vs["component_type"] = list(COMP_DF['component_type'].values)
G.vs["cost_fraction"] = list(COMP_DF['cost_fraction'].values)
G.vs["node_type"] = list(COMP_DF['node_type'].values)
G.vs["node_cluster"] = list(COMP_DF['node_cluster'].values)
G.vs["capacity"] = 1.0
G.vs["functionality"] = 1.0

for index, row in NODE_CONN_DF.iterrows():
    G.add_edge(row['Orig'], row['Dest'],
               capacity=G.vs.find(row['Orig'])["capacity"],
               weight=row['Weight'],
               distance=row['Distance'])

#                    --------
# Network setup with NetworkX (for drawing graph)
#                    --------
X = nx.DiGraph()
for index, row in NODE_CONN_DF.iterrows():
    X.add_edge(row['Orig'], row['Dest'],
               capacity=row['Capacity'],
               weight=row['Weight'],
               distance=row['Distance'])

systemlayout.draw_sys_layout(X, COMP_DF, out_dir=OUTPUT_PATH,
                             graph_label="System Component Layout")

# -----------------------------------------------------------------------------
# List of tagged nodes with special roles:
sup_node_list = [str(k) for k in
                 list(COMP_DF.ix[COMP_DF['node_type'] == 'supply'].index)]
dep_node_list = [str(k) for k in
                 list(COMP_DF.ix[COMP_DF['node_type'] == 'dependency'].index)]
src_node_list = [k for (k, v)in X.in_degree().iteritems() if v == 0]
out_node_list = list(SYSOUT_SETUP.index.get_level_values('OutputNode'))

# -----------------------------------------------------------------------------
# Power output and economic loss calculations
# -----------------------------------------------------------------------------

PGA_str = [('%0.3f' % np.float(x)) for x in hazard_intensity_vals]

cptype = {}
cptype_ds_edges = {}
for comp in nodes_all:
    cptype[comp] = comp_dict['component_type'][comp]
    cptype_ds_edges[cptype[comp]] =\
        sorted(fragdict['damage_ratio'][cptype[comp]].values())

###############################################################################
# component_resp_dict = {k:{c:{} for c in nodes_all} for k in PGA_str}
tp_cp = []
for x in comps_costed:
    tp_cp.extend(
        ((x, 'loss_mean'), (x, 'loss_std'), (x, 'func_mean'), (x, 'func_std'))
    )
mindex = pd.MultiIndex.from_tuples(tp_cp, names=['component_id', 'response'])
component_resp_df = pd.DataFrame(index=mindex, columns=[PGA_str])
component_resp_dict = component_resp_df.to_dict()

###############################################################################
# simulation of damage of each component

calculated_output_array = np.zeros((NUM_SAMPLES, num_hazard_pts))
economic_loss_array = np.zeros_like(calculated_output_array)

comp_loss_array = np.zeros((NUM_SAMPLES, num_hazard_pts))
comp_loss_dict = {c: np.zeros((NUM_SAMPLES, num_hazard_pts)) for c in nodes_all}

# Record output for:
# <samples> vs <hazard parameter index> vs <time step index>
output_array_given_recovery = np.zeros((NUM_SAMPLES, num_hazard_pts, num_time_steps))

rnd = stats.uniform.rvs(loc=0, scale=1, size=(NUM_SAMPLES, num_elements))
np.save(os.path.join(RAW_OUTPUT_DIR, 'rnd_samples_x_elements.npy'), rnd)

sys_output_dict = {k: {o: 0 for o in out_node_list} for k in PGA_str}

# List of output values at output_nodes:
sys_output_list_given_pga = {k: np.zeros((NUM_SAMPLES, len(out_node_list)))
                             for k in PGA_str}

comp_dsix_given_pga = {k: np.zeros((NUM_SAMPLES, len(nodes_all)))
                       for k in PGA_str}

ids_comp_vs_haz = {p: np.zeros((NUM_SAMPLES, num_elements)) for p in PGA_str}


def calc_loss_arrays(rnd,
                     calculated_output_array,
                     economic_loss_array,
                     output_array_given_recovery):

    print("\nCalculating system response to hazard transfer parameters...")

    for idxPGA, PGA in enumerate(PGA_str):
        print(" {0:3d}  out of {1:3d}".format(idxPGA+1, num_hazard_pts))

        # compute pe and determine ds for each component
        ids_comp = np.zeros((NUM_SAMPLES, num_elements), dtype=int)

        # index of damage state of components: from 0 to nds+1
        for j, comp in enumerate(nodes_all):
            ids_comp[:, j] = np.sum(
                cal_pe_ds(comp, float(PGA), comp_dict, fragdict)
                > rnd[:, j][:, np.newaxis], axis=1
                )

        # comp_loss_dict[comp] = np.zeros((NUM_SAMPLES,num_hazard_pts))

        component_loss_tmp = {c: [] for c in nodes_all}
        component_func_tmp = {c: [] for c in nodes_all}

        # system output and economic loss
        for i in range(NUM_SAMPLES):
            loss_list_all_comp = []
            cp_func = []
            cp_func_given_time = []

            for j, comp_name in enumerate(nodes_all):
                # ........................................................
                comp_type = comp_dict['component_type'][comp_name]
                ids = ids_comp[i, j]     # index for component damage state
                ds = dmg_states[ids]   # damage state name
                cf = comp_dict['cost_fraction'][comp_name]
                dr = fragdict['damage_ratio'][comp_type][ds]
                fn = fragdict['functionality'][comp_type][ds]
                loss = dr * cf
                loss_list_all_comp.append(loss)

                # ........................................................
                # component functionality for calculated damage state:
                cp_func.append(fn)
                cp_func_given_time.append(
                    calc_recov_time_given_comp_ds(comp_name, ids))

                comp_loss_dict[comp_name][i, idxPGA] = loss
                component_loss_tmp[comp_name].append(loss)
                component_func_tmp[comp_name].append(fn)
                # ........................................................

            economic_loss_array[i, idxPGA] = sum(loss_list_all_comp)

            outputlist = compute_output_given_ds(cp_func)
            calculated_output_array[i, idxPGA] = sum(outputlist)

            sys_output_list_given_pga[PGA][i, :] = outputlist

            # restoration status of components over the range of time
            # (num elements X num specified time units)
            cp_func_given_time = np.array(cp_func_given_time)
            for t in range(num_time_steps):
                output_array_given_recovery[i, idxPGA, t]\
                    = sum(compute_output_given_ds(cp_func_given_time[:, t]))

        ids_comp_vs_haz[PGA] = ids_comp

        for j, comp_name in enumerate(nodes_all):
            component_resp_dict[PGA][(comp_name, 'loss_mean')]\
                = np.mean(component_loss_tmp[comp_name])

            component_resp_dict[PGA][(comp_name, 'loss_std')]\
                = np.std(component_loss_tmp[comp_name])

            component_resp_dict[PGA][(comp_name, 'func_mean')]\
                = np.mean(component_func_tmp[comp_name])

            component_resp_dict[PGA][(comp_name, 'func_std')]\
                = np.std(component_func_tmp[comp_name])

            component_resp_dict[PGA][(comp_name, 'num_failures')]\
                = np.mean(ids_comp[:, j] >= (len(dmg_states) - 1))

        for onx, onode in enumerate(out_node_list):
            sys_output_dict[PGA][onode]\
                = np.mean(sys_output_list_given_pga[PGA][:, onx])

calc_loss_arrays(rnd, calculated_output_array, economic_loss_array, output_array_given_recovery)

idshaz = os.path.join(RAW_OUTPUT_DIR, 'ids_comp_vs_haz.pickle')
with open(idshaz, 'w') as handle:
    pickle.dump(ids_comp_vs_haz, handle)

# -----------------------------------------------------------------------------
# System output file (for given hazard transfer parameter value)
# -----------------------------------------------------------------------------

sys_output_df = pd.DataFrame(sys_output_dict)
sys_output_df.index.name = 'Output Nodes'

outfile_sysoutput = os.path.join(OUTPUT_PATH, 'system_output_vs_hazparam.csv')
sys_output_df.to_csv(outfile_sysoutput, sep=',', index_label=['Output Nodes'])

# -----------------------------------------------------------------------------
# Loss calculations by Component Type
# -----------------------------------------------------------------------------
tp_ct = []
for x in cp_types_costed:
    tp_ct.extend(
        ((x, 'loss_mean'), (x, 'loss_std'), (x, 'loss_tot'),
         (x, 'func_mean'), (x, 'func_std'))
    )

mindex = pd.MultiIndex.from_tuples(tp_ct, names=['component_type', 'response'])
comptype_resp_df = pd.DataFrame(index=mindex, columns=[PGA_str])
comptype_resp_dict = comptype_resp_df.to_dict()

for p in PGA_str:
    for ct in cp_types_costed:

        comptype_resp_dict[p][(ct, 'loss_mean')] =\
            np.mean([component_resp_dict[p][(cn, 'loss_mean')]
                     for cn in cpmap[ct]])

        comptype_resp_dict[p][(ct, 'loss_tot')] =\
            np.sum([component_resp_dict[p][(cn, 'loss_mean')]
                    for cn in cpmap[ct]])

        comptype_resp_dict[p][(ct, 'loss_std')] =\
            np.mean([component_resp_dict[p][(cn, 'loss_std')]
                     for cn in cpmap[ct]])

        comptype_resp_dict[p][(ct, 'func_mean')] =\
            np.mean([component_resp_dict[p][(cn, 'func_mean')]
                     for cn in cpmap[ct]])

        comptype_resp_dict[p][(ct, 'func_std')] =\
            np.mean([component_resp_dict[p][(cn, 'func_std')]
                     for cn in cpmap[ct]])

        comptype_resp_dict[p][(ct, 'num_failures')] =\
            np.mean([component_resp_dict[p][(cn, 'num_failures')]
                     for cn in cpmap[ct]])


# -----------------------------------------------------------------------------
# System Fragility & Probability of Exceedence
# -----------------------------------------------------------------------------
#
#   Damage state boundaries for Economic Loss calculations
#
#   None: < 0.01,
#   Slight: 0.01 to 0.15,
#   Moderate: 0.15 to 0.4,
#   Extensive: 0.4 to 0.8,
#   Complete: 0.8 to 1.0
#
# -----------------------------------------------------------------------------

sys_dmg_states = ['DS0 None',
                  'DS1 Slight',
                  'DS2 Moderate',
                  'DS3 Extensive',
                  'DS4 Complete']

sys_ds_bounds = [0.01, 0.15, 0.4, 0.8, 1.0]
# sys_ds_bounds = [0.04, 0.30, 0.75, 0.99, 1.10]

# --- System fragility ---
sys_frag = np.zeros_like(economic_loss_array, dtype=int)
for j in range(num_hazard_pts):
    for i in range(NUM_SAMPLES):
        # system output and economic loss
        sys_frag[i, j] = np.sum(economic_loss_array[i, j] > sys_ds_bounds)

# --- Probability of Exceedence ---
pe_sys_econloss = np.zeros((len(sys_dmg_states), num_hazard_pts))
for j in range(num_hazard_pts):
    for i in range(len(sys_dmg_states)):
        pe_sys_econloss[i, j] = np.sum(sys_frag[:, j] >= i)/float(NUM_SAMPLES)

# -----------------------------------------------------------------------------
# For Probability of Exceedence calculations based on component failures
# -----------------------------------------------------------------------------
#
#   Damage state boundaries for Component Type Failures (Substations) are
#   based on HAZUS MH MR3, p 8-66 to 8-68
#
# -----------------------------------------------------------------------------

cp_classes_in_system = list(np.unique(COMP_DF['component_class'].tolist()))
cp_class_map = {k: [] for k in cp_classes_in_system}
for k, v in comp_dict['component_class'].iteritems():
    cp_class_map[v].append(k)

if SYSTEM_CLASS == 'Substation':
    uncosted_classes = ['JUNCTION POINT', 'SYSTEM INPUT', 'SYSTEM OUTPUT',
                        'Generator', 'Bus', 'Lightning Arrester']
    ds_lims_compclasses = {
        'Disconnect Switch':   [0.05, 0.40, 0.70, 0.99, 1.00],
        'Circuit Breaker':     [0.05, 0.40, 0.70, 0.99, 1.00],
        'Current Transformer': [0.05, 0.40, 0.70, 0.99, 1.00],
        'Voltage Transformer': [0.05, 0.40, 0.70, 0.99, 1.00],
        'Power Transformer':   [0.05, 0.40, 0.70, 0.99, 1.00],
        'Control Building':    [0.06, 0.30, 0.75, 0.99, 1.00]
    }

    cp_classes_costed = \
        [x for x in cp_classes_in_system if x not in uncosted_classes]

    # --- System fragility - Based on Failure of Component Classes ---
    comp_class_failures = {cc: np.zeros((NUM_SAMPLES, num_hazard_pts))
                           for cc in cp_classes_costed}
    comp_class_frag = {cc: np.zeros((NUM_SAMPLES, num_hazard_pts))
                       for cc in cp_classes_costed}
    for j, PGA in enumerate(PGA_str):
        for i in range(NUM_SAMPLES):
            for compclass in cp_classes_costed:
                for c in cp_class_map[compclass]:
                    comp_class_failures[compclass][i, j] += \
                        ids_comp_vs_haz[PGA][i, nodes_all.index(c)]
                comp_class_failures[compclass][i, j] /= \
                    len(cp_class_map[compclass])
                comp_class_frag[compclass][i, j] = \
                    np.sum(comp_class_failures[compclass][i, j]
                           > ds_lims_compclasses[compclass])

    # --- Probability of Exceedence - Based on Failure of Component Classes ---
    pe_sys_cpfailrate = np.zeros((len(sys_dmg_states), num_hazard_pts))
    for p in range(num_hazard_pts):
        for d in range(len(sys_dmg_states)):
            ds_ss_ix = []
            for compclass in cp_classes_costed:
                ds_ss_ix.append(np.sum(comp_class_frag[compclass][:, p] >= d)
                                / float(NUM_SAMPLES))
            pe_sys_cpfailrate[d, p] = np.median(ds_ss_ix)

    # --- Save prob exceedance data as npy ---
    np.save(os.path.join(RAW_OUTPUT_DIR, 'pe_sys_cpfailrate.npy'),
            pe_sys_cpfailrate)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if SYSTEM_CLASS == 'Power Station (Coal Fired)':
    uncosted_classes = ['JUNCTION POINT', 'SYSTEM INPUT', 'SYSTEM OUTPUT']
    ds_lims_compclasses = {
        'Boiler':                       [0.0, 0.05, 0.40, 0.70, 1.00],
        'Control Building':             [0.0, 0.05, 0.40, 0.70, 1.00],
        'Emission Management':          [0.0, 0.05, 0.40, 0.70, 1.00],
        'Fuel Delivery and Storage':    [0.0, 0.05, 0.40, 0.70, 1.00],
        'Fuel Movement':                [0.0, 0.05, 0.40, 0.70, 1.00],
        'Generator':                    [0.0, 0.05, 0.40, 0.70, 1.00],
        'SYSTEM OUTPUT':                [0.0, 0.05, 0.40, 0.70, 1.00],
        'Stepup Transformer':           [0.0, 0.05, 0.40, 0.70, 1.00],
        'Turbine':                      [0.0, 0.05, 0.40, 0.70, 1.00],
        'Water System':                 [0.0, 0.05, 0.40, 0.70, 1.00]
    }

    cp_classes_costed = \
        [x for x in cp_classes_in_system if x not in uncosted_classes]

# -----------------------------------------------------------------------------
# Validate damage ratio of the system

exp_damage_ratio = np.zeros((len(nodes_all), num_hazard_pts))
for l, PGA in enumerate(hazard_intensity_vals):
    # compute expected damage ratio
    for j, comp_name in enumerate(nodes_all):
        pb = pe2pb(cal_pe_ds(comp_name, PGA, comp_dict, fragdict))
        comp_type = comp_dict['component_type'][comp_name]
        dr = np.array([fragdict['damage_ratio'][comp_type][ds]
                       for ds in dmg_states])
        cf = comp_dict['cost_fraction'][comp_name]
        loss_list = dr * cf
        exp_damage_ratio[j, l] = np.sum(pb * loss_list)

# -----------------------------------------------------------------------------
# Time to Restoration of Full Capacity
# -----------------------------------------------------------------------------

threshold = 0.99
required_time = []

for j in range(num_hazard_pts):
    cpower = \
        np.mean(output_array_given_recovery[:, j, :], axis=0) /\
        nominal_production
    temp = cpower > threshold
    if sum(temp) > 0:
        required_time.append(np.min(restoration_time_range[temp]))
    else:
        required_time.append(RESTORE_TIME_MAX)

# -----------------------------------------------------------------------------
# *** Saving vars ***
# -----------------------------------------------------------------------------

if SAVE_VARS_NPY:
    np.save(os.path.join(RAW_OUTPUT_DIR, 'economic_loss_array.npy'),
            economic_loss_array)
    np.save(os.path.join(RAW_OUTPUT_DIR, 'calculated_output_array.npy'),
            calculated_output_array)
    np.save(os.path.join(RAW_OUTPUT_DIR, 'output_array_given_recovery.npy'),
            output_array_given_recovery)
    np.save(os.path.join(RAW_OUTPUT_DIR, 'exp_damage_ratio.npy'),
            exp_damage_ratio)
    np.save(os.path.join(RAW_OUTPUT_DIR, 'sys_frag.npy'),
            sys_frag)
    np.save(os.path.join(RAW_OUTPUT_DIR, 'required_time.npy'),
            required_time)
    np.save(os.path.join(RAW_OUTPUT_DIR, 'pe_sys_econloss.npy'),
            pe_sys_econloss)

# -----------------------------------------------------------------------------
# Write analytical outputs to file
# -----------------------------------------------------------------------------

# ---< Output File >--- summary output ---
outfile_sys_response = os.path.join(OUTPUT_PATH, 'system_response.csv')
out_cols = ['PGA', 'Economic Loss', 'Mean Output', 'Days to Full Recovery']
outdat = {out_cols[0]: hazard_intensity_vals,
          out_cols[1]: np.mean(economic_loss_array, axis=0),
          out_cols[2]: np.mean(calculated_output_array, axis=0),
          out_cols[3]: required_time}
df = pd.DataFrame(outdat)
df.to_csv(outfile_sys_response, sep=',', index=False, columns=out_cols)

# ---< Output File >--- response of each COMPONENT to hazard ---
outfile_comp_resp = os.path.join(OUTPUT_PATH, 'component_response.csv')
component_resp_df = pd.DataFrame(component_resp_dict)
component_resp_df.index.names = ['component_id', 'response']
component_resp_df.to_csv(outfile_comp_resp, sep=',',
                         index_label=['component_id', 'response'])

# ---< Output File >--- mean loss of component ---
outfile_comp_loss = os.path.join(OUTPUT_PATH, 'component_meanloss.csv')
component_loss_df = component_resp_df.iloc[
    component_resp_df.index.get_level_values(1) == 'loss_mean']
component_loss_df.reset_index(level='response', inplace=True)
component_loss_df = component_loss_df.drop('response', axis=1)
component_loss_df.to_csv(outfile_comp_loss, sep=',',
                         index_label=['component_id'])

# ---< Output File >--- response of each COMPONENT TYPE to hazard ---
outfile_comptype_resp = os.path.join(OUTPUT_PATH, 'comp_type_response.csv')
comptype_resp_df = pd.DataFrame(comptype_resp_dict)
comptype_resp_df.index.names = ['component_type', 'response']
comptype_resp_df.to_csv(outfile_comptype_resp, sep=',',
                        index_label=['component_type', 'response'])

# ---< Output File >--- mean loss of component type ---
outfile_comptype_loss = os.path.join(OUTPUT_PATH, 'comp_type_meanloss.csv')
comptype_loss_df = comptype_resp_df.iloc[
    comptype_resp_df.index.get_level_values(1) == 'loss_mean']
comptype_loss_df.reset_index(level='response', inplace=True)
comptype_loss_df = comptype_loss_df.drop('response', axis=1)
comptype_loss_df.to_csv(outfile_comptype_loss, sep=',',
                        index_label=['component_type'])

# ---< Output File >--- mean failures for component types ---
outfile_comptype_failures = os.path.join(OUTPUT_PATH,
                                         'comp_type_meanfailures.csv')
comptype_failure_df = comptype_resp_df.iloc[
    comptype_resp_df.index.get_level_values(1) == 'num_failures']
comptype_failure_df.reset_index(level='response', inplace=True)
comptype_failure_df = comptype_failure_df.drop('response', axis=1)
comptype_failure_df.to_csv(outfile_comptype_failures, sep=',',
                           index_label=['component_type'])

# # ---< Output File >--- DataFrame with mean failures per component CLASS ---
# outfile_compclass_failures = os.path.join(OUTPUT_PATH,
#                                           'comp_class_meanfailures.csv')
# compclass_failure_df.to_csv(outfile_compclass_failures, sep=',',
#                         index_label=['component_class'])

# # ---< Output File >--- probabiility of exceedence per component
# outfile_comp_pe = os.path.join(OUTPUT_PATH, 'component_pe.csv')
# component_pe_df = pd.DataFrame(comp_pe)
# component_pe_df = component_pe_df.iloc[
#                     component_pe_df.index.get_level_values(1)!='DS0 None']
# component_pe_df.to_csv(outfile_comp_pe, sep=',')

# -----------------------------------------------------------------------------
# For plots
# -----------------------------------------------------------------------------

labels = dmg_states[1:]
markers = ['+', 'o', 's', '^', 'D', '$\\times$']
xt_step = 0.1   # x-tick spacing

# -----------------------------------------------------------------------------
# BOX PLOT :: MEAN ECONOMIC LOSS

fig = plt.figure(figsize=(9, 5), facecolor='white')
ax = fig.add_subplot(111, axisbg='white')
ax.spines['bottom'].set_position(('axes', -0.05))

ppl.boxplot(ax, economic_loss_array, widths=0.35)
xt_pos, xt_val = spl.calc_tick_pos(xt_step, hazard_intensity_vals, PGA_str,
                                   plot_type='box')

figfile = os.path.join(OUTPUT_PATH, 'fig_lossratio_boxplot.png')
spl.format_fig(
    ax,
    figtitle='Boxplot of loss ratio vs. PGA',
    figfile=figfile,
    x_lab=hazard_transfer_label,
    y_lab='Loss Ratio',
    x_tick_pos=xt_pos,
    x_tick_val=xt_val,
    x_grid=False,
    y_grid=True,
    x_margin=0.05,
    y_margin=None,
)

plt.close(fig)

# -----------------------------------------------------------------------------
# PLOT :: MEAN ECONOMIC LOSS VS SHAKING INTENSITY

fig = plt.figure(figsize=(9, 5), facecolor='white')
ax = fig.add_subplot(111, axisbg='white')

ax.plot(hazard_intensity_vals, np.mean(economic_loss_array, axis=0),
        label='Simulation', clip_on=False,
        linestyle='-', marker='o', markersize=5,
        markeredgecolor=spl.COLR_DARK2[2], color=spl.COLR_DARK2[2])

ax.plot(hazard_intensity_vals, np.sum(exp_damage_ratio, axis=0),
        label='Expected', clip_on=False,
        linestyle='-', marker='x', markersize=5,
        markeredgecolor=spl.COLR_DARK2[1], color=spl.COLR_DARK2[1])

xt_pos, xt_val = spl.calc_tick_pos(xt_step, hazard_intensity_vals, PGA_str,
                                   plot_type='line')

figfile = os.path.join(OUTPUT_PATH, 'fig_lossratio_lineplot.png')
spl.format_fig(
    ax,
    figtitle='Economic Loss Ratio',
    figfile=figfile,
    x_lab=hazard_transfer_label,
    y_lab='Loss Ratio',
    x_tick_pos=xt_pos,
    y_tick_pos=None,
    x_tick_val=xt_val,
    y_tick_val=None,
    x_grid=False,
    y_grid=True,
    x_margin=0.05,
    y_margin=None,
    add_legend=True,
    legend_title=' ',
)

plt.close(fig)

# -----------------------------------------------------------------------------
# PLOT :: PROBABILITY OF EXCEEDENCE : ECONOMIC LOSS

import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('paper')

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

for i in range(1, len(sys_dmg_states)):
    ax.plot(hazard_intensity_vals, pe_sys_econloss[i], label=sys_dmg_states[i],
            linestyle='--', linewidth=0.8, color=spl.COLR_DS[i], alpha=0.4,
            marker=markers[i], markersize=4, markeredgecolor=spl.COLR_DS[i],
            clip_on=False)

figfile = os.path.join(OUTPUT_PATH, 'fig_pr_ex_econloss.png')
figtitle = 'System Fragility Based on Economic Loss'
legend_title = 'Damage States'

xt_pos, xt_val = spl.calc_tick_pos(xt_step, hazard_intensity_vals, PGA_str,
                                   plot_type='line')
yt_pos = [0.00, 0.25, 0.50, 0.75, 1.00]

ax.grid(True, which="major", linestyle='-', linewidth=0.5)
spines_to_keep = ['bottom', 'left', 'top', 'right']
for spine in spines_to_keep:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_linewidth(0.7)

ax.set_xticks(xt_val)
ax.set_yticks(yt_pos)

ax.set_xlabel(hazard_transfer_label)
ax.set_ylabel('Pr [ $D_s > d_s$ | PGA ]')

ax.set_title(figtitle, loc='center', y=1.04, size=10)
ax.legend(title=legend_title, loc='upper left',
          ncol=1, bbox_to_anchor=(1.02, 1.0),
          frameon=0, prop={'size': 8})
ax.get_legend().get_title().set_fontsize('10')

plt.savefig(figfile, format='png', bbox_inches='tight', dpi=300)
plt.close(fig)

# -----------------------------------------------------------------------------
# PLOT :: PROBABILITY OF EXCEEDENCE : COMPONENT CLASS FAILURE RATES

if SYSTEM_CLASS == 'Substation':

    sns.set_style('whitegrid')
    sns.set_context('paper')

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    for i in range(1, len(sys_dmg_states)):
        ax.plot(hazard_intensity_vals, pe_sys_cpfailrate[i], label=sys_dmg_states[i],
                linestyle='--', linewidth=1, color=spl.COLR_DS[i], alpha=0.4,
                marker=markers[i], markersize=5,
                markeredgecolor=spl.COLR_DS[i], clip_on=False)

    figfile = os.path.join(OUTPUT_PATH, 'fig_pr_ex_compfailrate.png')
    figtitle = 'System Fragility Based on Component Failure Rates'
    legend_title = 'Damage States'

    xt_pos, xt_val = spl.calc_tick_pos(xt_step, hazard_intensity_vals, PGA_str,
                                       plot_type='line')
    yt_pos = [0.00, 0.25, 0.50, 0.75, 1.00]

    ax.grid(True, which="major", linestyle='-', linewidth=0.5)
    spines_to_keep = ['bottom', 'left', 'top', 'right']
    for spine in spines_to_keep:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.7)

    ax.set_xticks(xt_val)
    ax.set_yticks(yt_pos)

    ax.set_xlabel(hazard_transfer_label)
    ax.set_ylabel('Pr[ $D_s > d_s$ | PGA ]')

    ax.set_title(figtitle, loc='center', y=1.04, size=10)
    ax.legend(title=legend_title, loc='upper left',
              ncol=1, bbox_to_anchor=(1.02, 1.0),
              frameon=0, prop={'size': 8})

    plt.savefig(figfile, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)

# -----------------------------------------------------------------------------
# BOX PLOT :: mean output vs shaking intensity

fig = plt.figure(figsize=(9, 5), facecolor='white')
ax = fig.add_subplot(111, axisbg='white')
ax.spines['bottom'].set_position(('axes', -0.05))

ppl.boxplot(ax, calculated_output_array, widths=0.35)
xt_pos, xt_val = spl.calc_tick_pos(xt_step, hazard_intensity_vals, PGA_str,
                                   plot_type='box')

figfile = os.path.join(OUTPUT_PATH, 'fig_system_output.png')
spl.format_fig(
    ax,
    figtitle='% Mean Output vs Shaking Intensity',
    figfile=figfile,
    x_lab=hazard_transfer_label,
    y_lab='System Output',
    x_tick_pos=xt_pos,
    y_tick_pos=None,
    x_tick_val=xt_val,
    y_tick_val=None,
    x_grid=False,
    y_grid=True,
    x_margin=0.05,
    y_margin=None,
)

plt.close(fig)

# -----------------------------------------------------------------------------
# PLOT :: RECOVERY TIME TO FULL CAPACITY

fig = plt.figure(figsize=(9, 5), facecolor='white')
ax = fig.add_subplot(111, axisbg='white')

ppl.plot(ax, hazard_intensity_vals, required_time, label='Restoration Time',
         linestyle='-', color=spl.COLR_RDYLGN[2],
         marker='o', markeredgecolor=spl.COLR_RDYLGN[2],
         clip_on=False)

xt_pos, xt_val = spl.calc_tick_pos(xt_step, hazard_intensity_vals, PGA_str,
                                   plot_type='line')

figfile = os.path.join(OUTPUT_PATH, 'fig_restoration_time_vs_haz.png')
spl.format_fig(
    ax,
    figtitle='Required Time for Full Recovery',
    figfile=figfile,
    x_lab=hazard_transfer_label,
    y_lab='Restoration Time ('+TIME_UNIT+')',
    x_tick_pos=xt_pos,
    y_tick_pos=None,
    x_tick_val=xt_val,
    y_tick_val=None,
    # y_lim=[0, RESTORE_TIME_MAX],
    x_grid=False,
    y_grid=True,
    x_margin=0.05,
    y_margin=None,
    add_legend=True,
)

plt.close(fig)

# *****************************************************************************
# # Plot and save FRAGILITIES of all componements:
# plot_comp_frag(FRAGILITIES, costed_comptypes, hazard_intensity_vals, OUTPUT_PATH)

print("\nOutputs saved in: " + Fore.GREEN + OUTPUT_PATH)
