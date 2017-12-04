from __future__ import print_function
import os
import networkx as nx
import re
from networkx.readwrite.json_graph import node_link_data

# -----------------------------------------------------------------------------


class SystemTopology(object):

    orientation = "LR"          # Orientation of graph - Graphviz option
    connector_type = "spline"   # Connector appearance - Graphviz option
    clustering = False          # Cluster based on defined `node_cluster`

    out_file = "system_topology"
    graph_label = "System Component Topology"

    def __init__(self, infrastructure, scenario):
        self.infrastructure = infrastructure
        self.scenario = scenario
        self.gviz = ""  # placeholder for a pygraphviz agraph
        self.component_attr = {}  # Dict for system comp attributes
        self.out_dir = ""

        for comp_id in infrastructure.components.keys():
            self.component_attr[comp_id] = \
                vars(infrastructure.components[comp_id])

        if infrastructure.system_class.lower() in \
                ['potablewatertreatmentplant']:
            self.out_dir=scenario.output_path
            self.graph_label="Water Treatment Plant Component Topology"
            self.orientation="TB"
            self.connector_type="ortho"
            self.clustering=True
        else:
            self.out_dir=scenario.output_path
            self.graph_label="System Component Topology"
            self.orientation="LR"
            self.connector_type="spline"
            self.clustering=False


    def draw_sys_topology(self, viewcontext):
        """
        Draws the component configuration for a given infrastructure system.
        :param G: ipython graph object
        :param component_attr: dict of attributes of system components
        :param out_dir: location for savings outputs
        :param out_file: name of output file
        :param graph_label: label for the graph topology image
        :param orientation: orientation of the graph (default is top-to-bottom)
        :param connector_type: line connector type. Must be one of the types
            supported by Graphviz (i.e. 'spline', 'ortho', 'line', 'polyline',
            'curved')
        :param clustering: whether to create subgraphs based on `node_cluster`
            designated for components
        :return: saves a visual representation of the topology of system
            components in multiple formats (png, svg, and dot)
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Set up output file names & location

        if not self.out_dir.strip():
            output_path = os.getcwd()
        else:
            output_path = self.out_dir

        # strip away file ext and add our own
        fname = self.out_file.split(os.extsep)[0]

        if self.orientation.upper() not in ['TB', 'LR', 'RL', 'BT']:
            self.orientation = 'TB'

        if self.connector_type.lower() not in \
                ['spline', 'ortho', 'line', 'polyline', 'curved']:
            self.connector_type = 'spline'
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Draw graph using pygraphviz, and define general node & edge attributes
        G = self.infrastructure._component_graph.comp_graph
        graphml_file = os.path.join(output_path, fname + '.graphml')
        G.write_graphml(graphml_file)

        elist = G.get_edgelist()
        named_elist = []
        for tpl in elist:
            named_elist.append((G.vs[tpl[0]]['name'],
                                G.vs[tpl[1]]['name']))
        nxG = nx.DiGraph(named_elist)

        self.gviz = nx.nx_agraph.to_agraph(nxG)

        self.gviz.graph_attr.update(
            resolution=200,
            directed=True,
            labelloc="t",
            label='< <BR/>'+self.graph_label+'<BR/> >',
            rankdir=self.orientation,
            ranksep="1.0 equally",
            splines=self.connector_type,
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

        self.gviz.node_attr.update(
            shape="circle",
            style="rounded,filled",
            fixedsize="true",
            width=1.8,
            height=1.8,
            xlp="0, 0",
            color="royalblue3",  # gray14
            fillcolor="white",
            fontcolor="royalblue3",  # gray14
            penwidth=1.5,
            fontname="Helvetica-Bold",
            fontsize=18,
        )

        self.gviz.edge_attr.update(
            arrowhead="normal",
            arrowsize="1.0",
            style="bold",
            color="royalblue2",  # gray12
            penwidth=1.2,
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Customise nodes based on node type or defined clusters

        for node in self.component_attr.keys():
            label_mod = self.segment_long_labels(node, delims=['_', ' '])
            self.gviz.get_node(node).attr['label'] = label_mod

            if str(self.component_attr[node]['node_type']).lower() \
                    == 'supply':
                self.gviz.get_node(node).attr['label'] = \
                    self.segment_long_labels(node, maxlen=12,
                                             delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    label=self.gviz.get_node(node).attr['label'],
                    shape="rect",
                    style="rounded,filled",
                    fixedsize="true",
                    color="limegreen",
                    fillcolor="white",
                    fontcolor="limegreen",
                    penwidth=2.0,
                    height=1.2,
                    width=2.2,
                )

            if str(self.component_attr[node]['node_type']).lower() \
                    == 'sink':
                self.gviz.get_node(node).attr.update(
                    shape="doublecircle",
                    rank="sink",
                    penwidth=2.0,
                    color="orangered",  # royalblue3
                    fillcolor="white",
                    fontcolor="orangered",  # royalblue3
                )

            if str(self.component_attr[node]['node_type']).lower() \
                    == 'dependency':
                self.gviz.get_node(node).attr.update(
                    shape="circle",
                    rank="dependency",
                    penwidth=3.5,
                    color="orchid",
                    fillcolor="white",
                    fontcolor="orchid"
                )

        node_clusters = list(set([self.component_attr[id]['node_cluster']
                                  for id in self.component_attr.keys()]))
        if self.clustering == True:
            for cluster in node_clusters:
                grp = [k for k in self.component_attr.keys()
                       if self.component_attr[k]['node_cluster']==cluster]
                cluster = '_'.join(cluster.split())
                if  cluster.lower() not in ['none', '']:
                    cluster_name = 'cluster_'+cluster
                    rank = 'same'
                else:
                    cluster_name = ''
                    rank = ''
                self.gviz.add_subgraph(
                    nbunch=grp,
                    name=cluster_name,
                    style='invis',
                    label='',
                    clusterrank='local',
                    rank=rank,
                )

        if viewcontext == "as-built":
            self.gviz.write(os.path.join(output_path, fname + '.dot'))
            self.gviz.draw(os.path.join(output_path, fname + '.png'),
                format='png', prog='dot')

        self.gviz.draw(os.path.join(output_path, fname + '.svg'),
            format='svg', prog='dot', args='-Gsize=11,8\! -Gdpi=300')

        # nx.readwrite.json_graph.node_link_data(self.gviz,
        #                   os.path.join(output_path, fname + '.json'))

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    def msplit(self, string, delims):
        s = string
        for d in delims:
            rep = d + '\n'
            s = rep.join(x for x in s.split(d))
        return s

    def segment_long_labels(self, string, maxlen=7, delims=[]):
        if (not delims) and (len(string) > maxlen):
            return "\n".join(
                re.findall("(?s).{," + str(maxlen) + "}", string))[:-1]
        elif len(string) > maxlen:
            return self.msplit(string, delims)
        else:
            return string



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
