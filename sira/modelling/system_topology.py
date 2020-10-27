import os
import re

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

matplotlib.use('Agg')
plt.switch_backend('agg')

# from networkx.readwrite.json_graph import node_link_data


class SystemTopology(object):

    orientation = "LR"          # Orientation of graph - Graphviz option
    connector_type = "spline"   # Connector appearance - Graphviz option
    clustering = False          # Cluster based on defined `node_cluster`

    out_file = "system_topology"
    graph_label = "System Component Topology"

    def __init__(self, infrastructure, scenario):

        self.loc_attr = 'SYSTEM_COMPONENT_LOCATION_CONF'
        self.graph_label = "System Component Topology"

        self.infrastructure = infrastructure
        self.scenario = scenario
        self.gviz = ""  # placeholder for a pygraphviz agraph
        self.component_attr = {}  # Dict for system comp attributes
        self.out_dir = scenario.output_path
        self.node_position_meta = \
            self.infrastructure.system_meta[self.loc_attr]['value']

        for comp_id in list(infrastructure.components.keys()):
            self.component_attr[comp_id] = \
                vars(infrastructure.components[comp_id])

        if infrastructure.system_class.lower() in [
                'potablewatertreatmentplant', 'pwtp',
                'wastewatertreatmentplant', 'wwtp',
                'substation']:
            self.orientation = "TB"
            self.connector_type = "ortho"
            self.clustering = True
        elif infrastructure.system_class.lower() in \
                ['powerstation']:
            self.orientation = "LR"
            self.connector_type = "ortho"
            self.clustering = True
        else:
            self.orientation = "TB"
            self.connector_type = "polyline"
            self.clustering = False

        # Default drawing program
        self.drawing_prog = 'dot'

        # Overwrite default if node locations are defined
        if hasattr(infrastructure, 'system_meta'):
            if self.infrastructure.system_meta[self.loc_attr]['value']\
                    == 'defined':
                self.drawing_prog = 'neato'

    def draw_sys_topology(self, viewcontext):
        if self.infrastructure.system_class.lower() in ['substation']:
            self.draw_substation_topology(viewcontext)
        elif self.infrastructure.system_class.lower() in [
                "potablewatertreatmentplant", "pwtp",
                "wastewatertreatmentplant", "wwtp",
                "watertreatmentplant", "wtp"]:
            self.draw_wtp_topology(viewcontext)
        else:
            self.draw_generic_sys_topology(viewcontext)

    def draw_generic_sys_topology(self, viewcontext):
        """
        Draws the component configuration for a given infrastructure system.

        :param viewcontext: Option "as-built" indicates topology of system
        prior to hazard impact. Other options can be added to reflect
        post-impact system configuration and alternate designs.
        :return: generates and saves the system topology diagram in the
        following formats: (graphviz) dot, png, svg.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Set up output file names & location

        if not str(self.out_dir).strip():
            output_path = os.getcwd()
        else:
            output_path = self.out_dir

        # strip away file ext and add our own
        fname = self.out_file.split(os.extsep)[0]

        # Orientation of the graph (default is top-to-bottom):
        if self.orientation.upper() not in ['TB', 'LR', 'RL', 'BT']:
            self.orientation = 'TB'

        # `connector_type` refers to the line connector type. Must be one of
        # the types supported by Graphviz (i.e. 'spline', 'ortho', 'line',
        # 'polyline', 'curved')
        if self.connector_type.lower() not in \
                ['spline', 'ortho', 'line', 'polyline', 'curved']:
            self.connector_type = 'ortho'
        if str(self.node_position_meta).lower() == 'defined':
            self.drawing_prog = 'neato'
        else:
            self.drawing_prog = 'dot'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Draw graph using pygraphviz. Define general node & edge attributes.

        G = self.infrastructure._component_graph.digraph
        graphml_file = os.path.join(output_path, fname + '.graphml')
        G.write_graphml(graphml_file)

        elist = G.get_edgelist()
        named_elist = []
        for tpl in elist:
            named_elist.append((G.vs[tpl[0]]['name'],
                                G.vs[tpl[1]]['name']))
        G_nx = nx.DiGraph(named_elist)

        self.gviz = nx.nx_agraph.to_agraph(G_nx)

        default_node_color = "royalblue3"
        default_edge_color = "royalblue2"

        self.gviz.graph_attr.update(
            concentrate=False,
            resolution=300,
            directed=True,
            labelloc="t",
            label='< ' + self.graph_label + '<BR/><BR/> >',
            rankdir=self.orientation,
            splines=self.connector_type,
            center="true",
            forcelabels=True,
            fontname="Helvetica-Bold",
            fontcolor="#444444",
            fontsize=26,
            smoothing="graph_dist",
            pad=0.5,
            nodesep=1.5,
            sep=1.0,
            overlap="voronoi",
            overlap_scaling=1.0,
        )

        self.gviz.node_attr.update(
            shape="circle",
            style="rounded,filled",
            fixedsize="true",
            width=1.8,
            height=1.8,
            xlp="0, 0",
            color=default_node_color,  # gray14
            fillcolor="white",
            fontcolor=default_node_color,  # gray14
            penwidth=1.5,
            fontname="Helvetica-Bold",
            fontsize=22,
        )

        self.gviz.edge_attr.update(
            arrowhead="normal",
            arrowsize="1.0",
            style="bold",
            color=default_edge_color,  # gray12
            penwidth=1.8,
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Customise nodes based on node type or defined clusters

        for node in list(self.component_attr.keys()):
            label_mod = self.segment_long_labels(node, delims=['_', ' '])
            self.gviz.get_node(node).attr['label'] = label_mod

            if str(self.component_attr[node]['node_type']).lower() == 'supply':
                self.gviz.get_node(node).attr['label'] =\
                    self.segment_long_labels(node, maxlen=12, delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    shape="rect",
                    rank="supply",
                    style="rounded,filled",
                    fixedsize="true",
                    color="limegreen",
                    fillcolor="white",
                    fontcolor="limegreen",
                    penwidth=2.0,
                    height=1.2,
                    width=2.2,
                )

            if str(self.component_attr[node]['node_type']).lower() == 'sink':
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

            if str(self.component_attr[node]['node_type']).lower() \
                    == 'junction':
                self.gviz.get_node(node).attr.update(
                    shape="point",
                    width=0.5,
                    height=0.5,
                    penwidth=3.5,
                    color=default_node_color,
                )

        # Clustering: whether to create subgraphs based on `node_cluster`
        #             designated for components
        node_clusters = list(set([self.component_attr[k]['node_cluster']
                                  for k in list(self.component_attr.keys())]))
        if self.clustering:
            for cluster in node_clusters:
                grp = [k for k in list(self.component_attr.keys())
                       if self.component_attr[k]['node_cluster'] == cluster]
                if cluster:
                    cluster = '_'.join(cluster.split())
                    cluster_name = 'cluster_' + cluster
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

        for node in list(self.component_attr.keys()):
            pos_x = self.component_attr[node]['pos_x']
            pos_y = self.component_attr[node]['pos_y']
            if pos_x and pos_y:
                node_pos = str(pos_x) + "," + str(pos_y) + "!"
                self.gviz.get_node(node).attr.update(pos=node_pos)

        # self.gviz.layout(prog=self.drawing_prog)
        if viewcontext == "as-built":
            if self.drawing_prog == 'neato':
                draw_args = '-n -Gdpi=300'
            else:
                draw_args = '-Gdpi=300'
            self.gviz.write(os.path.join(output_path, fname + '_gv.dot'))
            self.gviz.draw(os.path.join(output_path, fname + '_dot.png'),
                           format='png', prog='dot',
                           args='-Gdpi=300')
            self.gviz.draw(os.path.join(output_path, fname + '.png'),
                           format='png', prog=self.drawing_prog,
                           args=draw_args)

        self.gviz.draw(os.path.join(output_path, fname + '.svg'),
                       format='svg',
                       prog=self.drawing_prog)

        # nx.readwrite.json_graph.node_link_data(self.gviz,
        #                   os.path.join(output_path, fname + '.json'))

    # ==========================================================================

    def draw_substation_topology(self, viewcontext):
        """
        Draws the component configuration for a substation.

        :param viewcontext: Option "as-built" indicates topology of system
        prior to hazard impact. Other options can be added to reflect
        post-impact system configuration and alternate designs.
        :return: generates and saves the system topology diagram in the
        following formats: (graphviz) dot, png, svg.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Set up output file names & location

        if not str(self.out_dir).strip():
            output_path = os.getcwd()
        else:
            output_path = self.out_dir

        # strip away file ext and add our own
        fname = self.out_file.split(os.extsep)[0]

        # Orientation of the graph (default is top-to-bottom):
        self.orientation = 'TB'

        # `connector_type` refers to the line connector type. Must be one of
        # ['spline', 'ortho', 'line', 'polyline', 'curved']
        self.connector_type = 'ortho'
        if str(self.node_position_meta).lower() == 'defined':
            self.drawing_prog = 'neato'
        else:
            self.drawing_prog = 'dot'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        G = self.infrastructure._component_graph.digraph
        graphml_file = os.path.join(output_path, fname + '.graphml')
        G.write_graphml(graphml_file)

        elist = G.get_edgelist()
        named_elist = []
        for tpl in elist:
            named_elist.append((G.vs[tpl[0]]['name'],
                                G.vs[tpl[1]]['name']))
        G_nx = nx.DiGraph(named_elist)

        self.gviz = nx.nx_agraph.to_agraph(G_nx)

        default_node_color = "royalblue3"
        default_edge_color = "royalblue2"

        self.gviz.graph_attr.update(
            directed=True,
            concentrate=False,
            resolution=300,
            orientation="portrait",
            labelloc="t",
            label='< ' + self.graph_label + '<BR/><BR/> >',
            bgcolor="white",
            rankdir=self.orientation,
            splines=self.connector_type,
            center="true",
            forcelabels=True,
            fontname="Helvetica-Bold",
            fontcolor="#444444",
            fontsize=30,
            pad=0.5,
            pack=False,
            sep="+20",
            smoothing="none",
            # smoothing="graph_dist",
            # ranksep="1.0 equally",
            # overlap=False,
            # overlap="voronoi",
            # overlap_scaling=1.0,
        )

        self.gviz.node_attr.update(
            shape="circle",
            style="rounded,filled",
            fixedsize="true",
            width=0.3,
            height=0.3,
            color=default_node_color,  # gray14
            fillcolor="white",
            fontcolor=default_node_color,  # gray14
            penwidth=1.5,
            fontname="Helvetica-Bold",
            fontsize=22,
        )

        self.gviz.edge_attr.update(
            arrowhead="normal",
            arrowsize="0.7",
            style="bold",
            color=default_edge_color,  # gray12
            penwidth=1.8,
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Clustering: whether to create subgraphs based on `node_cluster`
        #             designated for components
        node_clusters = list(set([self.component_attr[k]['node_cluster']
                                  for k in list(self.component_attr.keys())]))
        if self.clustering:
            for cluster in node_clusters:
                grp = [k for k in list(self.component_attr.keys())
                       if self.component_attr[k]['node_cluster'] == cluster]
                if cluster:
                    cluster = '_'.join(cluster.split())
                    cluster_name = 'cluster_' + cluster
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

        for node in list(self.component_attr.keys()):
            pos_x = self.component_attr[node]['pos_x']
            pos_y = self.component_attr[node]['pos_y']
            if pos_x and pos_y:
                node_pos = str(pos_x) + "," + str(pos_y) + "!"
                self.gviz.get_node(node).attr.update(pos=node_pos)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Customise nodes based on node type or defined clusters

        for node in list(self.component_attr.keys()):
            # label_mod = self.segment_long_labels(node, delims=['_', ' '])
            # self.gviz.get_node(node).attr['label'] = label_mod

            if str(self.component_attr[node]['node_type']).lower() == 'supply':
                self.gviz.get_node(node).attr['label'] =\
                    self.segment_long_labels(node, maxlen=10, delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    shape="rect",
                    rank="supply",
                    style="filled",
                    fixedsize="true",
                    color="limegreen",
                    fillcolor="white",
                    fontcolor="limegreen",
                    peripheries=2,
                    penwidth=1.5,
                    height=1.0,
                    width=2.0,
                )

            if str(self.component_attr[node]['node_type']).lower() == 'sink':
                self.gviz.get_node(node).attr['label'] =\
                    self.segment_long_labels(node, maxlen=7, delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    shape="doublecircle",
                    width=2,
                    height=2,
                    rank="sink",
                    penwidth=1.5,
                    color="orangered",  # royalblue3
                    fillcolor="white",
                    fontcolor="orangered",  # royalblue3
                )

            if str(self.component_attr[node]['node_type']).lower() \
                    == 'dependency':
                self.gviz.get_node(node).attr['label'] =\
                    self.segment_long_labels(node, maxlen=9, delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    shape="rect",
                    style="rounded",
                    width=2.1,
                    height=1.1,
                    rank="dependency",
                    penwidth=2.5,
                    color="orchid",
                    fillcolor="white",
                    fontcolor="orchid"
                )

            if str(self.component_attr[node]['node_type']).lower() \
                    == 'junction':
                self.gviz.get_node(node).attr.update(
                    shape="circle",
                    style="rounded,filled",
                    width=0.2,
                    height=0.2,
                    label="",
                    xlabel="",
                    color="#999999",
                    fillcolor="#BBBBBB",
                )

            if str(self.component_attr[node]['node_type']).lower() \
                    == 'transshipment':
                tmplabel = self.segment_long_labels(
                    node, maxlen=12, delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    fixedsize="true",
                    label="",
                    xlabel=tmplabel,
                )

            if str(self.component_attr[node]['component_class']).lower()\
                    == 'bus':
                # POSITION MUST BE IN POINTS for this to work
                buspos = self.gviz.get_node(node).attr['pos']
                if str(self.node_position_meta).lower() == 'defined':
                    poslist = [int(x.strip("!")) for x in buspos.split(",")]
                    buspos = str(poslist[0]) + "," + str(poslist[1] + 25) + "!"
                self.gviz.get_node(node).attr.update(
                    shape="rect",
                    penwidth=1.5,
                    width=2.3,
                    height=0.2,
                    label="",
                    xlabel=node,
                    xlp=buspos,
                )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Draw the graph

        if viewcontext == "as-built":
            if self.drawing_prog == 'neato':
                draw_args = '-n -Gdpi=300'
            else:
                draw_args = '-Gdpi=300'
            self.gviz.write(os.path.join(output_path, fname + '_gv.dot'))
            self.gviz.draw(os.path.join(output_path, fname + '_dot.png'),
                           format='png', prog='dot',
                           args='-Gdpi=300 -Gsize=8.27,11.69\!')  # noqa: W605

            self.gviz.draw(os.path.join(output_path, fname + '.png'),
                           format='png', prog=self.drawing_prog,
                           args=draw_args)

        self.gviz.draw(os.path.join(output_path, fname + '.svg'),
                       format='svg',
                       prog=self.drawing_prog)

    # ==========================================================================
    def draw_wtp_topology(self, viewcontext):
        """
        Draws the component configuration for a water treatment plant.

        :param viewcontext: Option "as-built" indicates topology of system
        prior to hazard impact. Other options can be added to reflect
        post-impact system configuration and alternate designs.
        :return: generates and saves the system topology diagram in the
        following formats: (graphviz) dot, png, svg.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Set up output file names & location

        if not str(self.out_dir).strip():
            output_path = os.getcwd()
        else:
            output_path = self.out_dir

        # strip away file ext and add our own
        fname = self.out_file.split(os.extsep)[0]

        # Orientation of the graph (default is top-to-bottom):
        self.orientation = 'TB'

        # `connector_type` refers to the line connector type. Must be one of
        # ['spline', 'ortho', 'line', 'polyline', 'curved']
        self.connector_type = 'ortho'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        G = self.infrastructure._component_graph.digraph
        graphml_file = os.path.join(output_path, fname + '.graphml')
        G.write_graphml(graphml_file)

        elist = G.get_edgelist()
        named_elist = []
        for tpl in elist:
            named_elist.append((G.vs[tpl[0]]['name'],
                                G.vs[tpl[1]]['name']))
        G_nx = nx.DiGraph(named_elist)

        self.gviz = nx.nx_agraph.to_agraph(G_nx)

        default_node_color = "royalblue3"
        default_edge_color = "royalblue2"

        self.gviz.graph_attr.update(
            directed=True,
            concentrate=False,
            resolution=300,
            orientation="portrait",
            labelloc="t",
            label='< ' + self.graph_label + '<BR/><BR/> >',
            bgcolor="white",
            rankdir=self.orientation,
            # ranksep="1.0 equally",
            splines=self.connector_type,
            center="true",
            forcelabels=True,
            fontname="Helvetica-Bold",
            fontcolor="#444444",
            fontsize=26,
            # smoothing="graph_dist",
            smoothing="none",
            pad=0.5,
            pack=False,
            sep="+20",
        )

        self.gviz.node_attr.update(
            shape="circle",
            style="filled",
            fixedsize="true",
            width=0.3,
            height=0.3,
            color=default_node_color,  # gray14
            fillcolor="white",
            fontcolor=default_node_color,  # gray14
            penwidth=1.5,
            fontname="Helvetica-Bold",
            fontsize=12,
        )

        self.gviz.edge_attr.update(
            arrowhead="normal",
            arrowsize="0.7",
            style="bold",
            color=default_edge_color,  # gray12
            penwidth=1.0,
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Clustering: whether to create subgraphs based on `node_cluster`
        #             designated for components
        node_clusters = list(set([self.component_attr[k]['node_cluster']
                                  for k in list(self.component_attr.keys())]))
        if self.clustering:
            for cluster in node_clusters:
                grp = [k for k in list(self.component_attr.keys())
                       if self.component_attr[k]['node_cluster'] == cluster]
                cluster = '_'.join(cluster.split())
                if cluster.lower() not in ['none', '']:
                    cluster_name = 'cluster_' + cluster
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

        for node in list(self.component_attr.keys()):
            pos_x = self.component_attr[node]['pos_x']
            pos_y = self.component_attr[node]['pos_y']
            if pos_x and pos_y:
                node_pos = str(pos_x) + "," + str(pos_y) + "!"
                self.gviz.get_node(node).attr.update(pos=node_pos)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Customise nodes based on node type or defined clusters

        for node in list(self.component_attr.keys()):
            # label_mod = self.segment_long_labels(node, delims=['_', ' '])
            # self.gviz.get_node(node).attr['label'] = label_mod

            if str(self.component_attr[node]['node_type']).lower() == 'supply':
                self.gviz.get_node(node).attr['label'] =\
                    self.segment_long_labels(node, maxlen=12, delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    shape="ellipse",
                    rank="supply",
                    fixedsize="true",
                    color="limegreen",
                    fillcolor="white",
                    fontcolor="limegreen",
                    penwidth=1.5,
                    width=1.5,
                    height=0.9,
                )

            if str(self.component_attr[node]['node_type']).lower() == 'sink':
                self.gviz.get_node(node).attr['label'] =\
                    self.segment_long_labels(node, maxlen=12, delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    shape="ellipse",
                    rank="sink",
                    color="orangered",  # royalblue3
                    fillcolor="white",
                    fontcolor="orangered",  # royalblue3
                    peripheries=2,
                    penwidth=1.5,
                    width=1.5,
                    height=0.9,
                )

            if str(self.component_attr[node]['node_type']).lower() \
                    == 'dependency':
                self.gviz.get_node(node).attr['label'] =\
                    self.segment_long_labels(node, maxlen=14, delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    shape="circle",
                    width=1.5,
                    height=1.5,
                    rank="dependency",
                    penwidth=2.5,
                    color="orchid",
                    fillcolor="white",
                    fontcolor="orchid"
                )

            if str(self.component_attr[node]['node_type']).lower() \
                    == 'junction':
                tmplabel =\
                    self.segment_long_labels(node, maxlen=10, delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    shape="point",
                    width=0.25,
                    height=0.25,
                    color="#777777",
                    fillcolor="#777777",
                    fontcolor="#777777",
                    label="",
                    xlabel=tmplabel,
                )

            if str(self.component_attr[node]['node_type']).lower() \
                    == 'transshipment':
                tmplabel = self.segment_long_labels(node, maxlen=14,
                                                    delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    width=0.3,
                    height=0.3,
                    fixedsize="true",
                    label="",
                    xlabel=tmplabel,
                )

            if str(self.component_attr[node]['component_class']).lower() in \
                    ['large tank',
                     'sedimentation basin',
                     'sedimentation basin - large']:
                self.gviz.get_node(node).attr['label'] =\
                    self.segment_long_labels(node, maxlen=15, delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    shape="rect",
                    penwidth=1.0,
                    width=2.5,
                    height=0.9,
                    xlabel="",
                )

            if str(self.component_attr[node]['component_class']).lower() in\
                    ['small tank',
                     'sedimentation basin - small',
                     'chlorination tank']:
                self.gviz.get_node(node).attr['label'] =\
                    self.segment_long_labels(node, maxlen=12, delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    shape="rect",
                    penwidth=1.0,
                    width=1.5,
                    height=0.9,
                    xlabel="",
                )

            if str(self.component_attr[node]['component_class']).lower()\
                    == 'chemical tank':
                tmplabel =\
                    self.segment_long_labels(node, maxlen=12, delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    # shape="cylinder",
                    shape="circle",
                    penwidth=1.0,
                    width=0.7,
                    height=0.7,
                    fixedsize="true",
                    label="",
                    xlabel=tmplabel,
                )

            if str(self.component_attr[node]['component_class']).lower() in \
                    ['building', 'small building']:
                tmplabel =\
                    self.segment_long_labels(node, maxlen=12, delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    style="rounded",
                    shape="box",
                    penwidth=2.0,
                    width=1.6,
                    height=0.9,
                    label=tmplabel,
                    xlabel="",
                )

            if str(self.component_attr[node]['component_class']).lower() in\
                    ['pump', 'pumps']:
                tmplabel =\
                    self.segment_long_labels(node, maxlen=12, delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    shape="hexagon",
                    penwidth=1.0,
                    width=0.5,
                    height=0.5,
                    fixedsize="true",
                    label="",
                    xlabel=tmplabel,
                )

            if str(self.component_attr[node]['component_class']).lower() in \
                    ['switchroom', 'power supply']:
                self.gviz.get_node(node).attr['label'] =\
                    self.segment_long_labels(node, maxlen=15, delims=['_', ' '])
                self.gviz.get_node(node).attr.update(
                    shape="rect",
                    style="rounded",
                    penwidth=1.0,
                    width=1.6,
                    height=0.9,
                    xlabel="",
                )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Draw the graph
        if viewcontext == "as-built":
            if self.drawing_prog == 'neato':
                draw_args = '-n -Gdpi=300'
            else:
                draw_args = '-Gdpi=300'
            self.gviz.draw(os.path.join(output_path, fname + '.png'),
                           format='png', prog=self.drawing_prog,
                           args=draw_args)
            self.gviz.write(os.path.join(output_path, fname + '_gv.dot'))
            self.gviz.draw(os.path.join(output_path, fname + '_dot.png'),
                           format='png', prog='dot',
                           args='-Gdpi=300 -Gsize=8.27,11.69\!')  # noqa: W605
        self.gviz.draw(os.path.join(output_path, fname + '.svg'),
                       format='svg',
                       prog=self.drawing_prog)

    # ==========================================================================

    def split_but_preserve_delims(self, string, delims):
        delimsplus = [x + "+" for x in map(re.escape, delims)]
        regexPattern = '|'.join(delimsplus)
        split_chars = [x for x in re.split(regexPattern, string) if x]
        delims_matched = re.findall(regexPattern, string)
        new_str = [split_chars[i] + delims_matched[i] for i in
                   range(len(delims_matched))]
        if len(delims_matched) < len(split_chars):
            new_str.append(split_chars[-1])
        return new_str

    def join_list_elems_given_len(self, str_list, num_str=20):
        i = 0
        newlist = []
        str_list.reverse()
        while str_list:
            elem = str_list.pop()
            # elemnxt = ""
            while str_list and len(elem + str_list[-1]) <= num_str:
                elemnxt = str_list.pop()
                if len(elem + elemnxt) <= num_str:
                    elem = elem + elemnxt
                    # elemnxt_used = True
                else:
                    # elemnxt_used = False
                    exit()
                i += 1
            newlist.append(elem)
            i += 1
        return newlist

    def segment_long_labels(self, string, maxlen=12, delims=[chr(0x20)]):
        if (not delims) and (len(string) > maxlen):
            str_list = re.findall("(?s).{1," + str(maxlen) + "}", string)
        elif len(string) > maxlen:
            str_list = self.split_but_preserve_delims(string, delims)
            str_list = self.join_list_elems_given_len(str_list, num_str=maxlen)
        else:
            return string
        # print(str_list)
        # print("\n".join(str_list))
        return "\n".join(str_list)
