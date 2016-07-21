import os
import networkx as nx
import pandas as pd


# ------------------------------------------------------------------------------

def msplit(string, delims):
    s = string
    for d in delims:
        rep = d + '\n'
        s = rep.join(x for x in s.split(d))
    return s


def segment_long_labels(string, chars=6, delims=[]):
    if (not delims) or \
            (len(string) > chars and len(string.split(delims[0])[0]) > 8):
        return '\n'.join(
            string[i:i + chars] for i in range(0, len(string), chars))
    elif len(string) > chars:
        return msplit(string, delims)
    else:
        return string


# ------------------------------------------------------------------------------

def draw_sys_layout(G, comp_df, out_dir="", out_file="system_layout",
                    graph_label="System Layout"):
    # --- Set up output file names & location ---
    if not out_dir.strip():
        output_path = os.getcwd()
    else:
        output_path = out_dir

    fn = out_file.split(os.extsep)[0]  # strip away file ext and add our own
    sys_config_dot = os.path.join(output_path, fn + '.dot')
    sys_config_img = os.path.join(output_path, fn + '.png')

    # --- Draw graph using pygraphviz ---

    A = nx.to_agraph(G)

    A.graph_attr.update(resolution=300,
                        size="10.25,7.75",
                        directed=True,
                        labelloc="t",
                        label=graph_label,
                        rankdir="LR",
                        splines="spline",
                        center="true",
                        forcelabels="true",
                        fontname="Helvetica-Bold",
                        fontcolor="#444444",
                        fontsize=18)

    A.node_attr.update(shape="circle",
                       style="rounded,filled",
                       fixedsize="true",
                       width=1.4,
                       height=1.4,
                       xlp="0, 0",
                       color="royalblue2",
                       fillcolor="white",
                       fontcolor="royalblue3",
                       penwidth=1.0,
                       fontname="Helvetica-Bold",
                       fontsize=14)

    A.edge_attr.update(arrowhead="normal",
                       arrowsize="1.0",
                       color="royalblue2",
                       penwidth=1.0)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    for node in comp_df.index.values.tolist():

        label_mod = segment_long_labels(node, delims=['_', ' '])
        A.get_node(node).attr['label'] = label_mod

        if str(comp_df.ix[node]['node_type']) == 'supply':
            A.get_node(node).attr.update(
                label="",
                xlabel=label_mod,
                shape='point',
                height=0.4,
                style='filled',
                fillcolor="royalblue2")

        if str(comp_df.ix[node]['node_type']) == 'sink':
            A.get_node(node).attr['shape'] = 'doublecircle'
            A.get_node(node).attr['rank'] = 'sink'

        if str(comp_df.ix[node]['node_type']) == 'dependency':
            A.get_node(node).attr.update(
                shape="circle",
                penwidth=2.0,
                color="Orchid",
                fontcolor="Orchid")

    for cluster in pd.unique(comp_df['node_cluster'].values):
        grp = comp_df[comp_df['node_cluster'] == cluster].index.values.tolist()
        A.add_subgraph(grp, rank='same')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    A.write(sys_config_dot)
    A.draw(sys_config_img, format='png', prog='dot')


# ------------------------------------------------------------------------------

def draw_wtp_layout(G, comp_df, out_dir="", out_file="system_layout",
                    graph_label="System Layout"):
    """Draws the component configutaion for water treatment plants."""
    # --- Set up output file names & location ---
    if not out_dir.strip():
        output_path = os.getcwd()
    else:
        output_path = out_dir

    fn = out_file.split(os.extsep)[0]  # strip away file ext and add our own
    sys_config_dot = os.path.join(output_path, fn + '.dot')
    sys_config_img = os.path.join(output_path, fn + '.png')

    # --- Draw graph using pygraphviz ---

    A = nx.to_agraph(G)

    A.graph_attr.update(resolution=300,
                        # size="10.25,7.75",
                        directed=True,
                        labelloc="t",
                        label=graph_label,
                        rankdir="TB",
                        splines="spline",
                        center="true",
                        forcelabels="true",
                        fontname="Helvetica-Bold",
                        fontcolor="#444444",
                        fontsize=18)

    A.node_attr.update(shape="circle",
                       style="rounded,filled",
                       fixedsize="true",
                       width=1.4,
                       height=1.4,
                       xlp="0, 0",
                       color="royalblue2",
                       fillcolor="white",
                       fontcolor="royalblue3",
                       penwidth=1.0,
                       fontname="Helvetica-Bold",
                       fontsize=14)

    A.edge_attr.update(arrowhead="normal",
                       arrowsize="1.0",
                       color="royalblue2",
                       penwidth=1.0)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    for node in comp_df.index.values.tolist():

        label_mod = segment_long_labels(node, delims=['_', ' '])
        A.get_node(node).attr['label'] = label_mod

        if str(comp_df.ix[node]['node_type']) == 'supply':
            A.get_node(node).attr.update(
                label="",
                xlabel=label_mod,
                shape='point',
                height=0.4,
                style='filled',
                fillcolor="royalblue2"
            )

        if str(comp_df.ix[node]['node_type']) == 'sink':
            A.get_node(node).attr['shape'] = 'doublecircle'
            A.get_node(node).attr['rank'] = 'sink'

        if str(comp_df.ix[node]['node_type']) == 'dependency':
            A.get_node(node).attr.update(
                shape="circle",
                penwidth=2.0,
                color="Orchid",
                fontcolor="Orchid"
            )

    for cluster in pd.unique(comp_df['node_cluster'].values):
        grp = comp_df[comp_df['node_cluster'] == cluster].index.values.tolist()
        A.add_subgraph(grp, rank='same')

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    A.write(sys_config_dot)
    A.draw(sys_config_img, format='png', prog='dot')


# ------------------------------------------------------------------------------

'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*** Required Development: Absolute Node Positioning ***
i.e. the ability to specify exact location of each node on the canvas.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

The implementation could use this advice from graphviz developer:
http://www.graphviz.org/content/set-positions-node#comment-1771

Most of the Graphviz layout algorithms ignore position information. Indeed,
setting initial positions doesn't fit well with what the algorithms are trying
to do. The general idea is to specify more abstract constraints and then let
the algorithm do its best. That said, neato and fdp do allow you to provide
initial position information. Simply set the pos attribute in your input graph.
For example,

graph G { abc [pos="200,300"] }

(If you run fdp or neato, use the -s flag to make sure the coordinates are
interpreted as point values. Also, if you provide positions,
you may find neato -Kmode=KK better.) For more information, see

http://www.graphviz.org/content/attrs#dpos
http://www.graphviz.org/content/attrs#kpoint

If you know all of the node positions, you can use:
neato -n or neato -n2 (without -s) to do edge routing followed by rendering.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
'''
