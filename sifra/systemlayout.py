from __future__ import print_function
import os
import sys
import networkx as nx
import pandas as pd
import re

# -----------------------------------------------------------------------------

def msplit(string, delims):
    s = string
    for d in delims:
        rep = d + '\n'
        s = rep.join(x for x in s.split(d))
    return s


def segment_long_labels(string, maxlen=7, delims=[]):
    if (not delims) and (len(string) > maxlen):
        # return '\n'.join(string[i:i+maxlen]
        #                  for i in range(0, len(string), maxlen))
        return "\n".join(re.findall("(?s).{,"+str(maxlen)+"}", string))[:-1]

    elif len(string) > maxlen:
        return msplit(string, delims)

    else:
        return string


# -----------------------------------------------------------------------------

def draw_sys_layout(G, comp_df,
                    out_dir="",
                    out_file="system_layout",
                    graph_label="System Layout",
                    orientation="TB",
                    connector_type="spline",
                    clustering=False):
    """Draws the component configuration for a given infrastructure system."""

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Set up output file names & location

    if not out_dir.strip():
        output_path = os.getcwd()
    else:
        output_path = out_dir

    fname = out_file.split(os.extsep)[0]  # strip away file ext and add our own
    sys_config_dot = os.path.join(output_path, fname + '.dot')

    if orientation.upper() not in ['TB', 'LR', 'RL', 'BT']:
        orientation = 'TB'

    if connector_type.lower() not in ['spline', 'ortho', 'line',
                                      'polyline', 'curved']:
        connector_type = 'spline'
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Draw graph using pygraphviz, and define general node & edge attributes

    A = nx.nx_agraph.to_agraph(G)

    A.graph_attr.update(
        resolution=200,
        directed=True,
        labelloc="t",
        label='< <BR/>'+graph_label+'<BR/> >',
        rankdir=orientation,
        ranksep="1.0 equally",
        splines=connector_type,
        center="true",
        forcelabels="true",
        pad=0.2,
        nodesep=0.4,
        fontname="Helvetica-Bold",
        fontcolor="#444444",
        fontsize=26,
        smoothing="graph_dist",
        concentrate="true",
    )

    A.node_attr.update(
        shape="circle",
        style="rounded,filled",
        fixedsize="true",
        width=1.5,
        height=1.5,
        xlp="0, 0",
        color="royalblue3",  # gray14
        fillcolor="white",
        fontcolor="royalblue3",  # gray14
        penwidth=1.5,
        fontname="Helvetica-Bold",
        fontsize=18,
    )

    A.edge_attr.update(
        arrowhead="normal",
        arrowsize="1.0",
        style="bold",
        color="royalblue2",  # gray12
        penwidth=1.2,
    )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Customise nodes based on node type or defined clusters

    for node in comp_df.index.values.tolist():

        label_mod = segment_long_labels(node, delims=['_', ' '])
        A.get_node(node).attr['label'] = label_mod

        if str(comp_df.ix[node]['node_type']).lower() == 'supply':
            A.get_node(node).attr['label'] = \
                segment_long_labels(node, maxlen=12, delims=['_', ' '])
            A.get_node(node).attr.update(
                label=A.get_node(node).attr['label'],
                shape="rect",
                style="rounded,filled",
                fixedsize="true",
                color="limegreen",
                fillcolor="white",
                fontcolor="limegreen",
                penwidth=2.0,
                height=1.1,
                width=2.0,
            )

        if str(comp_df.ix[node]['node_type']).lower() == 'sink':
            A.get_node(node).attr.update(
                shape="doublecircle",
                rank="sink",
                penwidth=2.0,
                color="orangered",  # royalblue3
                fillcolor="white",
                fontcolor="orangered",  # royalblue3
            )

        if str(comp_df.ix[node]['node_type']).lower() == 'dependency':
            A.get_node(node).attr.update(
                shape="circle",
                rank="dependency",
                penwidth=3.5,
                color="orchid",
                fillcolor="white",
                fontcolor="orchid"
            )

    if clustering == True:
        for cluster in pd.unique(comp_df['node_cluster'].values):
            grp = comp_df[comp_df['node_cluster'] == cluster].\
                index.values.tolist()
            cluster = '_'.join(cluster.split())
            if  cluster.lower() not in ['none', '']:
                cluster_name = 'cluster_'+cluster
                rank = 'same'
            else:
                cluster_name = ''
                rank = ''
            A.add_subgraph(
                nbunch=grp,
                name=cluster_name,
                style='invis',
                label='',
                clusterrank='local',
                rank=rank,
            )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    A.write(sys_config_dot)

    A.draw(os.path.join(output_path, fname + '.png'),
           format='png', prog='dot')

    A.draw(os.path.join(output_path, fname + '.svg'),
           format='svg', prog='dot', args='-Gsize=11,8\! -Gdpi=300')

# -----------------------------------------------------------------------------

def main():

    from sifraclasses import Scenario, PowerStation, PotableWaterTreatmentPlant

    SETUPFILE = sys.argv[1]
    discard = {}
    config = {}

    exec (open(SETUPFILE).read(), discard, config)

    print("Setting up objects...")
    FacilityObj = eval(config["SYSTEM_CLASS"])
    sc = Scenario(SETUPFILE)
    fc = FacilityObj(SETUPFILE)
    # Define input files, output location, scenario inputs
    # SYS_CONFIG_FILE = os.path.join(scn.input_path, sysobj.sys_config_file_name)

    print("Initiating drawing network model schematic...")
    fc.network.network_setup(fc)
    print("Drawing complete.\n")

if __name__=="__main__":
    main()


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
