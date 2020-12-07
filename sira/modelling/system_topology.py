import re
import sys
import inspect
import warnings
import itertools
from pathlib import Path

import numpy as np
import graphviz as gviz


# =====================================================================================

def split_but_preserve_delims(string, delims):
    delimsplus = [x + "+" for x in map(re.escape, delims)]
    regexPattern = '|'.join(delimsplus)
    split_chars = [x for x in re.split(regexPattern, string) if x]
    delims_matched = re.findall(regexPattern, string)
    new_str = [split_chars[i] + delims_matched[i] for i in
               range(len(delims_matched))]
    if len(delims_matched) < len(split_chars):
        new_str.append(split_chars[-1])
    return new_str


def join_list_elems_given_len(str_list, num_str=20):
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


def segment_long_labels(string, maxlen=10, delims=[chr(0x20)]):
    if (not delims) and (len(string) > maxlen):
        str_list = re.findall("(?s).{1," + str(maxlen) + "}", string)
    elif len(string) > maxlen:
        str_list = split_but_preserve_delims(string, delims)
        str_list = join_list_elems_given_len(str_list, num_str=maxlen)
    else:
        return string

    return "\n".join(str_list)

# =====================================================================================


class SystemTopologyGenerator(object):
    substation_tags = [
        'substation', 'switching_station'
    ]
    powerstation_tags = [
        'powerstation', 'power_station',
        'power_generator', 'power_generation'
    ]
    wtp_tags = [
        "potablewatertreatmentplant", "pwtp",
        "wastewatertreatmentplant", "wwtp",
        "watertreatmentplant", "wtp"
    ]

    system_tags = dict(
        SS=[
            'substation', 'switching_station'
        ],
        PS=[
            'powerstation', 'power_station',
            'power_generator', 'power_generation'
        ],
        WTP=[
            'potablewatertreatmentplant', 'pwtp',
            'wastewatertreatmentplant', 'wwtp',
            'watertreatmentplant', 'wtp'
        ]
    )

    def __init__(self, infrastructure, output_dir):
        pass

    def __new__(self, infrastructure, output_dir, *args):

        self.topoargs = [infrastructure, output_dir]
        system_class = infrastructure.system_class.lower()

        sys_tag = "Generic"
        for key, value_list in self.system_tags.items():
            if system_class in value_list:
                sys_tag = key
        topo_cls = eval('SystemTopology_' + sys_tag)

        clsmembers = inspect.getmembers(
            sys.modules[__name__],
            lambda member: inspect.isclass(member) and member.__module__ == __name__)
        clsmembers = [x[0] for x in clsmembers]
        print(clsmembers)

        return topo_cls(*self.topoargs)

        # if system_class in self.powerstation_tags:
        #     return SystemTopology_PS(*self.topoargs)

        # elif system_class in self.substation_tags:
        #     return SystemTopology_SS(*self.topoargs)

        # elif system_class in self.wtp_tags:
        #     return SystemTopology_WTP(*self.topoargs)

        # else:
        #     return SystemTopology_Generic(*self.topoargs)


class SystemTopology_Generic(object):

    out_file = "system_topology"
    graph_label = "System Component Topology"

    def __init__(self, infrastructure, output_dir):

        self.loc_attr = 'SYSTEM_COMPONENT_LOCATION_CONF'
        self.graph_label = "System Component Topology"
        self.graph = {}  # placeholder for the system's graph representation

        self.infrastructure = infrastructure
        self.component_attr = {}  # Dict for system comp attributes
        self.output_path = output_dir if Path(str(output_dir)).is_dir() else Path.cwd()

        self.node_position_meta = \
            self.infrastructure.system_meta[self.loc_attr]['value']
        for comp_id in list(infrastructure.components.keys()):
            self.component_attr[comp_id] = \
                vars(infrastructure.components[comp_id])

        self.primary_node_types = ['supply', 'sink', 'dependency', 'junction']

        # ---------------------------------------------------------------------
        # Orientation of the graph (default is top-to-bottom):
        # orientation_options = ['TB', 'LR', 'RL', 'BT']
        # edge_type refers to the line connector type in Graphviz
        # connector_options = ['spline', 'ortho', 'line', 'polyline', 'curved']

        # Default drawing parameters
        self.orientation = "TB"
        self.layout_engine = "dot"
        self.clustering = False

        if str(self.node_position_meta).lower() == 'defined':
            self.edge_type = 'ortho'
        else:
            self.edge_type = 'spline'

        self.set_core_attributes()

        # ---------------------------------------------------------------------
        # Overwrite default if node locations are defined
        if hasattr(infrastructure, 'system_meta'):
            if self.infrastructure.system_meta[self.loc_attr]['value'] == 'defined':
                self.layout_engine = 'neato'

    # ================================================================================

    def set_core_attributes(self):

        # Default colours
        default_node_color = "royalblue3"
        default_edge_color = "royalblue2"

        # Default GRAPH attributes
        self.graph_attr_dict = dict(
            orientation='portrait',
            directed='True',
            concentrate='False',
            resolution='300',
            labelloc="t",
            label='< ' + self.graph_label + '<BR/><BR/> >',
            bgcolor="white",
            splines=str(self.edge_type),
            center="True",
            forcelabels="True",
            fontname="Helvetica-Bold",
            fontcolor='#444444',
            fontsize='24',
            pack="False",
            rankdir=str(self.orientation),
            overlap='False',
            sep='+5',
            # nodesep='0.5',
        )

        # Default EDGE attributes
        self.attr_edges_default = dict(
            arrowhead="normal",
            arrowsize="1.0",
            style="bold",
            color=str(default_edge_color),
            penwidth="1.8",
        )

        # ----------------------------------------------------------------------------
        # Default attributes for primary NODE types

        self.attr_nodes = {}

        self.attr_nodes['default'] = dict(
            shape="circle",
            style="filled",
            fixedsize="true",
            width="1.9",
            height="1.9",
            xlp="0, 0",
            color=str(default_node_color),
            fillcolor="white",
            fontcolor="#555555",
            penwidth="1.5",
            fontname="Helvetica-Bold",
            fontsize="20",
            margin="0.2,0.1"
        )

        self.attr_nodes['supply'] = dict(
            shape="rect",
            rank="supply",
            style="rounded,filled",
            fixedsize="true",
            color="limegreen",
            fillcolor="white",
            fontcolor="limegreen",
            penwidth="2.0",
            width="2.4",
            height="1.2",
        )

        self.attr_nodes['sink'] = dict(
            shape="doublecircle",
            width="2",
            height="2",
            rank="sink",
            penwidth="2.0",
            color="orangered",
            fillcolor="white",
            fontcolor="orangered",
        )

        self.attr_nodes['dependency'] = dict(
            shape="rect",
            style="rounded, filled",
            rank="dependency",
            penwidth="3.5",
            width="2.2",
            height="1.0",
            color="orchid",
            fontcolor="orchid",
            fillcolor="white",
        )

        self.attr_nodes['junction'] = dict(
            shape="point",
            width="0.3",
            height="0.3",
            label="",
            xlabel="",
            color="#777777",
            fillcolor="#777777",
            fontcolor="#777777",
        )

    # ================================================================================

    def write_graphs_to_file(self, file_name,
                             dpi=300, canvas_size='',
                             engines=['dot', 'neato']):

        # ---------------------------------------------------------------------
        paper_sizes = {
            'A0': '33.1,46.8!', 'A1': '23.4,33.1!', 'A2': '16.5,23.4!',
            'A3': '11.7,16.5!', 'A4': '8.3,11.7!', 'A5': '5.8,8.3!'
        }

        # ---------------------------------------------------------------------
        # Check that valid layout engine names are supplied

        engines_valid = ['dot', 'neato', 'fdp', 'sfdp']
        engines_invalid = list(np.setdiff1d(engines, engines_valid))
        engines_matched = list(set(engines_valid).intersection(set(engines)))
        if engines_matched:
            engines = engines_matched
        else:
            engines = ['dot']
        if engines_invalid:
            warnings.warn(f"Invalid layout engines supplied: {engines_invalid}")
            warnings.warn(f"Using these layout engines: {engines}")

        # ---------------------------------------------------------------------
        # Update DPI value, if provided

        try:
            arg_res = int(str(dpi))
        except ValueError:
            print(f"Value of dpi must be an integer: {dpi}")
        if int(dpi) <= 0:
            raise ValueError("Value of dpi must be a non-zero positive integer.")
        arg_res = str(dpi)

        # ---------------------------------------------------------------------
        # Define output image size

        if canvas_size in paper_sizes.keys():
            arg_imgsize = paper_sizes[canvas_size]
        else:
            arg_imgsize = ''

        # ---------------------------------------------------------------------
        # Apply node clustering, if defined in config

        if self.clustering and self.layout_engine not in ['neato', 'fdp', 'sfdp']:
            self.apply_node_clustering()

        # Save graph in plain Graphviz dot format
        self.graph.save(
            filename=file_name + '.gv',
            directory=Path(self.output_path))

        # ---------------------------------------------------------------------
        # Save images in multiple formats

        img_formats = {'png': 'bitmap', 'pdf': 'vector'}
        for eng, fmt in itertools.product(engines, img_formats.keys()):
            x = arg_res if img_formats[fmt] == 'bitmap' else ''
            self.graph.graph_attr.update(size=arg_imgsize, dpi=x)  # noqa: W605
            self.graph.engine = eng
            self.graph.render(
                filename=file_name + '_' + eng,
                directory=Path(self.output_path),
                format=fmt,
                cleanup=True)

    # ================================================================================

    def draw_sys_topology(self):

        # 'output_path' is required and set at object initialisation
        fname = Path(self.out_file).stem

        # Default drawing parameters
        self.clustering = False
        self.configure_sys_topology()
        self.graph.attr(rankdir="LR", splines=self.edge_type)

        # Customise layout engines to be used?
        # Default engines: ['dot', 'neato']
        self.write_graphs_to_file(fname)

    # ================================================================================

    def build_graph_structure(self):

        # ---------------------------------------------------------------------
        # Set up output file names & location

        fname = Path(self.out_file).stem

        # ---------------------------------------------------------------------
        # Build the graph representing system topology, and
        # Define general node & edge attributes.
        G = self.infrastructure._component_graph.digraph
        elist = G.get_edgelist()
        named_elist = []
        for tpl in elist:
            named_elist.append((G.vs[tpl[0]]['name'], G.vs[tpl[1]]['name']))

        # ---------------------------------------------------------------------
        # # Generate dot language representation of graph using networkx & pydot
        # G_nx = nx.DiGraph(named_elist)
        # G_dot = nx.nx_pydot.to_pydot(G_nx).to_string()
        # self.graph = gviz.Source(G_dot, name='system_component_topology')

        # ---------------------------------------------------------------------
        # If required define custom system-wide graph attributes here
        # ...

        # ---------------------------------------------------------------------
        # Generate dot language representation of graph using python-graphviz
        self.graph = gviz.Digraph(
            name='system_component_topology',
            filename=fname, engine=self.layout_engine,
            graph_attr=self.graph_attr_dict,
            node_attr=self.attr_nodes['default'],
            edge_attr=self.attr_edges_default
        )
        self.graph.edges(named_elist)

    # ================================================================================

    def apply_node_clustering(self):
        # Clustering: whether to create subgraphs based on `node_cluster`
        # designated for components

        node_clusters = list(set(
            [self.component_attr[key]['node_cluster']
             for key in list(self.component_attr.keys())]
        ))

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
            with self.graph.subgraph(name=cluster_name) as sg:
                for n in grp:
                    sg.node(n)
                sg.graph_attr.update(
                    name=cluster_name,
                    style='invis',
                    label='',
                    clusterrank='local',
                    rank=rank)

    # ================================================================================

    def configure_sys_topology(self):
        """
        Draws the component configuration for a given infrastructure system.

        :output: generates and saves the system topology diagram in the
        following formats: (graphviz) dot, png, pdf.
        """

        self.build_graph_structure()

        # ---------------------------------------------------------------------
        # Customise node attributes based on node type or defined clusters

        # node_specific_attrs = {}
        for nid in list(self.component_attr.keys()):

            # Position assignment for nodes
            pos_x = self.component_attr[nid]['pos_x']
            pos_y = self.component_attr[nid]['pos_y']
            node_pos = ''
            if pos_x and pos_y:
                node_pos = str(pos_x) + "," + str(pos_y) + "!"

            # Segment long labels to fit within nodes spaces
            label_mod = segment_long_labels(nid, delims=['_', ' '])

            # Custom attribute assignment based on node types
            node_type = str(self.component_attr[nid]['node_type']).lower()
            if node_type not in self.primary_node_types:
                node_type = 'default'
            # node_specific_attrs = eval('self.attr_nodetype_' + node_type)
            # self.graph.node(nid, label_mod, pos=node_pos, **node_specific_attrs)
            self.graph.node(nid, label=label_mod, pos=node_pos,
                            **self.attr_nodes[node_type])

        # ---------------------------------------------------------------------
        return self.graph

# =====================================================================================


class SystemTopology_PS(SystemTopology_Generic):

    def __init__(self, *args):
        super(SystemTopology_PS, self).__init__(*args)

    def draw_sys_topology(self):
        fname = Path(self.out_file).stem
        self.clustering = False
        self.configure_sys_topology()
        self.graph.attr(rankdir="LR", splines="spline")
        self.write_graphs_to_file(fname, dpi=300, engines=['dot', 'neato'])

# =====================================================================================


class SystemTopology_SS(SystemTopology_Generic):

    def __init__(self, *args):
        super(SystemTopology_SS, self).__init__(*args)

    def draw_sys_topology(self):
        fname = Path(self.out_file).stem
        self.clustering = True
        self.configure_sys_topology()

        self.graph.attr(splines="ortho")
        self.write_graphs_to_file(fname, dpi=300, engines=['dot'])

        self.graph.attr(splines="line")
        self.write_graphs_to_file(fname, dpi=300, engines=['neato'])

    def configure_sys_topology(self):
        """
        Draws the component configuration for a given infrastructure system.

        :output: generates and saves the system topology diagram in the
        following formats: (graphviz) dot, png, pdf.
        """

        self.graph_attr_dict.update(
            orientation="portrait",
            rankdir="TB",
            splines=self.edge_type,
            fontsize="30",
            sep="+20",
            overlap="false",
        )
        self.build_graph_structure()

        # ---------------------------------------------------------------------
        # Customise node attributes based on node type or defined clusters

        self.attr_nodes['default'].update(
            shape="circle",
            style="rounded,filled",
            width="0.3",
            height="0.3",
            fontsize="22",
        )

        self.attr_nodes['bus'] = self.attr_nodes['default'].copy()
        self.attr_nodes['bus'].update(
            shape="rect",
            penwidth="1.5",
            width="2.3",
            height="0.2",
        )

        self.primary_node_types.append('bus')

        # node_specific_attrs = {}
        for nid in list(self.component_attr.keys()):

            # Position assignment for nodes
            pos_x = self.component_attr[nid]['pos_x']
            pos_y = self.component_attr[nid]['pos_y']
            node_pos = ''
            if pos_x and pos_y:
                node_pos = str(pos_x) + "," + str(pos_y) + "!"

            # Segment long labels to fit within nodes spaces
            label_mod = segment_long_labels(nid, delims=['_', ' '])

            # Custom attribute assignment based on node types
            node_type = str(self.component_attr[nid]['node_type']).lower()
            if node_type not in self.primary_node_types:
                node_type = 'default'

            # node_specific_attrs = eval('self.attr_nodetype_' + node_type)

            if node_type == 'bus':
                if str(self.node_position_meta).lower() == 'defined':
                    poslist = [int(x.strip("!")) for x in node_pos.split(",")]
                    buspos = str(poslist[0]) + "," + str(poslist[1] + 25) + "!"
                    self.graph.node(
                        nid, label='', xlabel=label_mod, xlp=buspos,
                        **self.attr_nodes[node_type])
                else:
                    self.graph.node(
                        nid, label='', xlabel=label_mod,
                        **self.attr_nodes[node_type])

            elif node_type == 'default':
                self.graph.node(
                    nid, label='', xlabel=label_mod, pos=node_pos,
                    **self.attr_nodes[node_type])

            elif node_type == 'junction':
                self.graph.node(
                    nid, label='', xlabel='', pos=node_pos,
                    **self.attr_nodes[node_type])

            else:
                self.graph.node(
                    nid, label=label_mod, pos=node_pos,
                    **self.attr_nodes[node_type])

        # ---------------------------------------------------------------------
        # Apply node clustering, if defined in config

        if self.clustering and self.layout_engine not in ['neato', 'fdp', 'sfdp']:
            self.apply_node_clustering()
        # ---------------------------------------------------------------------
        return self.graph

# =====================================================================================


class SystemTopology_WTP(SystemTopology_Generic):

    def __init__(self, *args):
        super(SystemTopology_WTP, self).__init__(*args)

    def draw_sys_topology(self):
        fname = Path(self.out_file).stem
        self.clustering = False
        self.configure_sys_topology()
        self.graph.attr(rankdir="TB", splines="ortho")
        self.write_graphs_to_file(fname, dpi=300, engines=['dot', 'neato'])

    def configure_sys_topology(self):
        """
        Draws the component configuration for a given infrastructure system.

        :output: generates and saves the system topology diagram in the
        following formats: (graphviz) dot, png, pdf.
        """

        self.graph_attr_dict.update(
            orientation="portrait",
            rankdir="TB",
            pad="0.5",
            sep="+20",
            pack="False",
            smoothing="none",
            overlap="voronoi",
        )
        self.build_graph_structure()

        # ---------------------------------------------------------------------
        # Customise node attributes based on type of infrastructure system

        self.attr_nodes['default'].update(
            shape="circle",
            style="rounded,filled",
            margin="20.0",
            # width="0.3",
            # height="0.3",
            # fontsize="15",
        )

        # ---------------------------------------------------------------------
        # Apply custom node attributes based on node type

        # node_specific_attrs = {}
        for nid in list(self.component_attr.keys()):

            # Position assignment for nodes
            pos_x = self.component_attr[nid]['pos_x']
            pos_y = self.component_attr[nid]['pos_y']
            node_pos = ''
            if pos_x and pos_y:
                node_pos = str(pos_x) + "," + str(pos_y) + "!"

            # Segment long labels to fit within nodes spaces
            label_mod = segment_long_labels(nid, maxlen=10, delims=['_', ' '])

            # Custom attribute assignment based on node types
            node_type = str(self.component_attr[nid]['node_type']).lower()
            component_class = str(self.component_attr[nid]['component_class']).lower()

            if node_type not in self.primary_node_types:
                node_type = 'default'

            if node_type in ['supply', 'source']:
                self.graph.node(
                    nid, label_mod, pos=node_pos, **self.attr_nodes['supply'])

            elif node_type in ['sink', 'consumer', 'output']:
                self.graph.node(
                    nid, label_mod, pos=node_pos, **self.attr_nodes['sink'])

            elif node_type in ['dependency']:
                self.graph.node(
                    nid, label_mod, pos=node_pos, **self.attr_nodes['dependency'])

            elif node_type in ['junction']:
                self.graph.node(
                    nid, label="", pos=node_pos, **self.attr_nodes['junction'])

            elif node_type in ['transshipment']:
                self.graph.node(
                    nid, label=label_mod, xlabel="", pos=node_pos,
                    **self.attr_nodes['default'])

            elif component_class in [
                    'large tank', 'large basin',
                    'sedimentation basin',
                    'sedimentation basin - large']:
                attr_nodetype_largebasin = self.attr_nodes['default'].copy()
                attr_nodetype_largebasin.update(
                    shape="rect", penwidth="1.0",
                    width="2.5", height="0.9")
                self.graph.node(
                    nid, label=label_mod, xlabel="",
                    pos=node_pos, **attr_nodetype_largebasin)

            elif component_class in [
                    'small tank', 'small basin',
                    'sedimentation basin - small',
                    'chlorination tank']:
                attr_nodetype_smallbasin = self.attr_nodes['default'].copy()
                attr_nodetype_smallbasin.update(
                    shape="rect", penwidth="1.0",
                    width="2.0", height="0.9")
                self.graph.node(
                    nid, label=label_mod, xlabel="",
                    pos=node_pos, **attr_nodetype_smallbasin)

            elif component_class in ['chemical tank']:
                attr_nodetype_chemtank = self.attr_nodes['default'].copy()
                attr_nodetype_chemtank.update(
                    shape="circle", penwidth="1.0",
                    width="1.0", height="0.7", fixedsize="True")
                self.graph.node(
                    nid, label="", xlabel=label_mod,
                    pos=node_pos, **attr_nodetype_chemtank)

            elif component_class in ['building', 'small building']:
                attr_nodetype_building_sml = self.attr_nodes['default'].copy()
                attr_nodetype_building_sml.update(
                    shape="box", style="rounded", penwidth="2.0",
                    width="1.8", height="0.9")
                self.graph.node(
                    nid, label=label_mod, xlabel="",
                    pos=node_pos, **attr_nodetype_building_sml)

            elif component_class in ['pump', 'pumps']:
                attr_nodetype_pump = self.attr_nodes['default'].copy()
                attr_nodetype_pump.update(
                    shape="hexagon", penwidth="1.0",
                    width="0.5", height="0.5", fixedsize="True")
                self.graph.node(
                    nid, label="", xlabel=label_mod,
                    pos=node_pos, **attr_nodetype_pump)

            elif component_class in ['switchroom', 'power supply']:
                attr_nodetype_switchroom = self.attr_nodes['default'].copy()
                attr_nodetype_switchroom.update(
                    shape="rect", style="rounded", penwidth="1.0",
                    width="1.6", height="0.9")
                self.graph.node(
                    nid, label=label_mod, xlabel="",
                    pos=node_pos, **attr_nodetype_switchroom)

            else:
                self.graph.node(
                    nid, label=label_mod, xlabel="",
                    pos=node_pos, **self.attr_nodes['default'])

        # ---------------------------------------------------------------------
        # Apply node clustering, if defined in config

        if self.clustering and self.layout_engine not in ['neato', 'fdp', 'sfdp']:
            self.apply_node_clustering()
        # ---------------------------------------------------------------------
        return self.graph

# =====================================================================================
