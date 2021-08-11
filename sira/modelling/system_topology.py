import re
import sys
import inspect
import warnings
import itertools
from pathlib import Path

import numpy as np
import networkx as nx

# from numpy.core.numeric import NaN


# =====================================================================================

def is_number(val):
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


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
        while str_list and len(elem + str_list[-1]) <= num_str:
            elemnxt = str_list.pop()
            if len(elem + elemnxt) <= num_str:
                elem = elem + elemnxt
            else:
                exit()
            i += 1
        newlist.append(elem)
        i += 1
    return newlist


def segment_long_labels(string, delims, maxlen=10):
    if delims in ['', 'NA', None]:
        delims = [chr(0x20)]
    if (not delims) and (len(string) > maxlen):
        str_list = re.findall("(?s).{1," + str(maxlen) + "}", string)
    elif len(string) > maxlen:
        str_list = split_but_preserve_delims(string, delims)
        str_list = join_list_elems_given_len(str_list, num_str=maxlen)
    else:
        return string

    # mod = "\n".join(str_list)
    mod = "<" + "<BR/>".join(str_list) + ">"
    return mod

# =====================================================================================


class SystemTopologyGenerator(object):

    system_tags = dict()

    # If newer classes are required to be defined, please add their
    # 'tags' and system names in the dict below.
    system_tags['SS'] = [
        'substation', 'switching_station'
    ]
    system_tags['PS'] = [
        'powerstation', 'power_station',
        'power_generator', 'power_generation'
    ]
    system_tags['WTP'] = [
        'potablewatertreatmentplant', 'pwtp',
        'wastewatertreatmentplant', 'wwtp',
        'watertreatmentplant', 'wtp',
        'potablewaterpumpstation'
    ]

    def __new__(self, infrastructure, output_dir):
        topoargs = [infrastructure, output_dir]
        system_class = infrastructure.system_class.lower()
        target_cls = self._get_target_class(self, system_class)
        return target_cls(*topoargs)

    def _get_target_class(self, system_name):

        # sys_tag = "Generic"
        # for key, value_list in system_tags.items():
        #     if system_name in value_list:
        #         sys_tag = key
        # topo_cls = eval('SystemTopology_' + sys_tag)

        # Identify the correct class based on the name provided
        if system_name.lower() in self.system_tags['PS']:
            topo_cls = SystemTopology_PS
        elif system_name.lower() in self.system_tags['SS']:
            topo_cls = SystemTopology_SS
        elif system_name.lower() in self.system_tags['WTP']:
            topo_cls = SystemTopology_WTP
        else:
            topo_cls = SystemTopology_Generic

        # Get list of all locally defined classes
        clsmembers = inspect.getmembers(
            sys.modules[__name__],
            lambda member: inspect.isclass(member) and member.__module__ == __name__)
        clsmembers = [x[0] for x in clsmembers]

        # Check that target class exists
        if str(topo_cls.__name__) not in clsmembers:
            raise NameError(
                f"The class {topo_cls.__name__} is not defined",
                f"in the module {__name__}")

        return topo_cls


class SystemTopology_Generic(object):

    out_file = "system_topology"

    def __init__(self, infrastructure, output_dir):

        self.loc_attr = 'SYSTEM_COMPONENT_LOCATION_CONF'
        self.graph_label = "System Component Topology"
        self.agraph = ""  # placeholder for the system's graph representation

        self.infrastructure = infrastructure
        self.component_attr = {}  # Dict for system comp attributes
        self.output_path = output_dir if Path(str(output_dir)).is_dir() else Path.cwd()

        self.node_position_meta \
            = self.infrastructure.system_meta[self.loc_attr]['value']
        for comp_id in list(infrastructure.components.keys()):
            self.component_attr[comp_id] \
                = vars(infrastructure.components[comp_id])

        # ---------------------------------------------------------------------
        # Orientation of the graph (default is top-to-bottom):
        # orientation_options = ['TB', 'LR', 'RL', 'BT']
        # edge_type refers to the line connector type in Graphviz
        # connector_options = ['spline', 'ortho', 'line', 'polyline', 'curved']
        self.primary_node_types = ['supply', 'sink', 'dependency', 'junction']
        self.engines_valid = ['dot', 'neato', 'fdp', 'sfdp']

        # ---------------------------------------------------------------------
        # Default drawing parameters

        self.orientation = "TB"
        self.clustering = False

        if str(self.node_position_meta).lower() == 'defined':
            self.edge_type = "ortho"
            self.layout_engine = "neato"
        else:
            self.edge_type = "spline"
            self.layout_engine = "dot"

        self.set_core_attributes()

    # ================================================================================

    def set_core_attributes(self):

        # Default colours
        self.default_node_color = "royalblue3"
        self.default_edge_color = "royalblue2"

        # Default GRAPH attributes
        self.graph_attr_dict = dict(
            orientation='portrait',
            directed='True',
            concentrate='False',
            resolution='300',
            labelloc="t",
            label=f"< {self.graph_label} <BR/><BR/> >",
            bgcolor="white",
            splines=str(self.edge_type),
            center="True",
            forcelabels="True",
            fontname="Helvetica-Bold",
            fontcolor="#444444",
            fontsize="24",
            pack="False",
            rankdir=str(self.orientation),
            overlap='False',
            pad='0.5',
            packmode='node',
            mode="sgd",
            maxiter="200",
            sep='+2',
            # size='11.7,16.5',  # IMPORTANT: this sets the max canvas size (inches)
        )

        # Default EDGE attributes
        self.attr_edges_default = dict(
            arrowhead="normal",
            arrowsize="1.0",
            style="bold",
            color=str(self.default_edge_color),
            penwidth="2.0",
            fontsize="30",
        )

        # Default attributes for primary NODE types
        self.attr_nodes = {}

        self.attr_nodes['default'] = dict(
            shape="circle",
            style="filled",
            width="2.5",
            height="2.5",
            fixedsize="True",
            color=str(self.default_node_color),
            fillcolor="white",
            fontcolor=str(self.default_node_color),
            penwidth="2",
            fontname="Helvetica-Bold",
            fontsize="30",
            margin="0.2,0.1"
        )

        self.attr_nodes['supply'] = dict(
            shape="rect",
            style="rounded,filled",
            width="4",
            height="1.8",
            rank="supply",
            fixedsize="False",
            color="limegreen",
            fillcolor="white",
            fontcolor="limegreen",
            fontsize="30",
            penwidth="2.0",
        )

        self.attr_nodes['sink'] = dict(
            shape="doublecircle",
            width="2.4",
            height="2.4",
            rank="sink",
            penwidth="2.0",
            color="orangered",
            fillcolor="white",
            fontcolor="orangered",
            fontsize="30",
        )

        self.attr_nodes['dependency'] = dict(
            shape="rect",
            width="3.5",
            height="1.8",
            style="rounded, filled",
            rank="dependency",
            penwidth="3.0",
            color="orchid",
            fontcolor="orchid",
            fillcolor="white",
            fontsize="30",
        )

        self.attr_nodes['junction'] = dict(
            shape="point",
            width="0.4",
            height="0.4",
            color="#888888",
            fillcolor="#AAAAAA",
            fontcolor="#888888",
            fontsize="30",
        )

    # ================================================================================

    def build_graph_structure(self):

        # ---------------------------------------------------------------------
        # Set up output file names & location
        # fname = Path(self.out_file).stem

        # ---------------------------------------------------------------------
        # Build the graph representing system topology, and
        # Define general node & edge attributes.

        G = self.infrastructure.get_component_graph()

        elist = G.get_edgelist()
        named_elist = []
        for tpl in elist:
            named_elist.append((G.vs[tpl[0]]['name'], G.vs[tpl[1]]['name']))

        G_nx = nx.DiGraph(named_elist)
        self.agraph = nx.nx_agraph.to_agraph(G_nx)

        # ---------------------------------------------------------------------

        self.agraph.graph_attr.update(**self.graph_attr_dict)
        self.agraph.node_attr.update(**self.attr_nodes['default'])
        self.agraph.edge_attr.update(**self.attr_edges_default)

    # ================================================================================

    def update_appearance_attributes(self):

        pass

    # ================================================================================

    def write_graphs_to_file(self, file_name,
                             dpi=300, paper_size=None,
                             engines=''):

        # ---------------------------------------------------------------------
        paper_size_dict = {
            'A0': '33.1,46.8!', 'A1': '23.4,33.1!', 'A2': '16.5,23.4!',
            'A3': '11.7,16.5!', 'A4': '8.3,11.7!', 'A5': '5.8,8.3!'
        }

        # ---------------------------------------------------------------------
        # Check that valid layout engine names are supplied

        # if not engines:
        #     if str(self.node_position_meta).lower() == 'defined':
        #         engines = ['dot', 'neato']
        #     else:
        #         engines = ['dot']

        if not engines:
            engines = ['dot', 'neato']

        engines_invalid = list(np.setdiff1d(engines, self.engines_valid))
        engines_matched = list(set(self.engines_valid).intersection(set(engines)))

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

        canvas_size = '8.27,11.69'
        if paper_size in paper_size_dict.keys():
            canvas_size = paper_size_dict[paper_size]
        else:
            canvas_size = '11.7,16.5'

        # ---------------------------------------------------------------------
        # Apply node clustering, if defined in config

        if self.clustering and self.layout_engine not in ['neato', 'fdp', 'sfdp']:
            self.apply_node_clustering()

        # Save graph in plain Graphviz dot format
        self.agraph.write(Path(self.output_path, file_name + '_gv.dot'))

        # ---------------------------------------------------------------------
        # Save images in multiple formats

        img_formats = {'ps2': 'vector', 'pdf': 'vector', 'png': 'bitmap'}
        draw_args = f"-Gdpi={arg_res} -Gsize={canvas_size}!"  # noqa: W1401

        for eng, fmt in itertools.product(engines, img_formats.keys()):
            try:
                self.agraph.draw(
                    path=Path(self.output_path, f"{file_name}_{eng}.{fmt}"),
                    format=fmt, prog=eng, args=draw_args)
            except (OSError, AssertionError, ValueError, TypeError):
                warnings.warn(
                    "Graph rendering failed for: {}, {}".format(eng, fmt),
                    stacklevel=2)

        # try:
        #     self.apply_node_clustering()
        #     self.agraph.draw(
        #         path=Path(self.output_path, f"{file_name}_clustered_.png"),
        #         format='png', prog='dot',
        #         args=f"-Gdpi={arg_res} -Gsize={canvas_size}!")
        # except (AssertionError, ValueError, TypeError) as inst:
        #     print(inst)
        #     warnings.warn("Graph rendering node clustering failed!")

    # ================================================================================

    def draw_sys_topology(self):

        # 'output_path' is required and set at object initialisation
        fname = Path(self.out_file).stem

        # Default drawing parameters
        self.clustering = False
        self.configure_sys_topology()
        self.agraph.graph_attr["rankdir"] = "LR"
        self.agraph.graph_attr["splines"] = self.edge_type
        self.write_graphs_to_file(fname)

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
            self.agraph.add_subgraph(
                nbunch=grp,
                name=cluster_name,
                style='invis',
                label='',
                clusterrank='local',
                rank=rank,
            )

    # ================================================================================

    def configure_sys_topology(self):
        """
        Draws the component configuration for a given infrastructure system.

        :output: generates and saves the system topology diagram in the
        following formats: (graphviz) dot, png, pdf, ps2.
        """

        self.build_graph_structure()

        # ---------------------------------------------------------------------
        # Customise node attributes based on node type or defined clusters

        # node_specific_attrs = {}
        for node_id in list(self.component_attr.keys()):

            # Position assignment for nodes
            pos_x = self.component_attr[node_id]['pos_x']
            pos_y = self.component_attr[node_id]['pos_y']
            node_pos = ''
            if is_number(pos_x) and is_number(pos_y):
                node_pos = str(pos_x) + "," + str(pos_y) + "!"

            # Segment long labels to fit within nodes spaces
            label_mod = segment_long_labels(node_id, delims=['_', ' '])

            # Custom attribute assignment based on node types
            node_type = str(self.component_attr[node_id]['node_type']).lower()
            if node_type not in self.primary_node_types:
                node_type = 'default'

            self.agraph.get_node(node_id).attr.update(
                label=label_mod, pos=node_pos,
                **self.attr_nodes[node_type]
            )

        return self.agraph

# =====================================================================================


class SystemTopology_PS(SystemTopology_Generic):

    def __init__(self, *args):
        super(SystemTopology_PS, self).__init__(*args)
        self.graph_label = "Power Station Topology"

    def draw_sys_topology(self):
        fname = Path(self.out_file).stem
        self.clustering = False
        self.configure_sys_topology()
        self.agraph.graph_attr["rankdir"] = "LR"
        self.write_graphs_to_file(fname, dpi=300, engines=['dot', 'neato'])

# =====================================================================================


class SystemTopology_SS(SystemTopology_Generic):

    def __init__(self, *args):
        super(SystemTopology_SS, self).__init__(*args)
        self.graph_label = "Electric Substation Topology"

    def draw_sys_topology(self):
        fname = Path(self.out_file).stem
        self.clustering = False
        self.build_graph_structure()
        self.configure_sys_topology()

        self.agraph.graph_attr.update(rankdir="TB", splines="spline")
        self.write_graphs_to_file(fname, dpi=300, engines=['dot'])

        self.agraph.graph_attr.update(rankdir="TB", splines="ortho")
        self.write_graphs_to_file(fname, dpi=600, engines=['neato'])

    def configure_sys_topology(self):
        """
        Draws the component configuration for a given infrastructure system.

        :output: generates and saves the system topology diagram in the
        following formats: (graphviz) dot, png, pdf, ps2.
        """

        # ----------------------------------------------------------------------------
        self.agraph.graph_attr.update(
            orientation="portrait",
            concentrate="False",
            rankdir="TB",
            center="True",
            forcelabels="True",
            fontsize="50",
            pad="0.1",
            pack="False",
            sep="+1",
            overlap="False",
            smoothing="none",
            ratio="auto",
            scale="1.0",
        )

        self.agraph.edge_attr.update(
            arrowhead="normal",
            arrowsize="5",
            penwidth="10",
            color=str(self.default_edge_color),
        )

        # ---------------------------------------------------------------------
        # Customise node attributes based on node type or defined clusters

        NODE_FONTSIZE = 33

        self.attr_nodes['default'].update(
            shape="circle",
            style="rounded,filled",
            width="0.5",
            height="0.5",
            penwidth="4",
            color=str(self.default_node_color),
            fontsize=str(NODE_FONTSIZE),
            fontcolor=str(self.default_node_color),
        )

        self.attr_nodes['supply'].update(
            shape="rect",
            rank="supply",
            style="rounded, filled",
            peripheries="1",
            width="3.5",
            height="1.8",
            penwidth="4.0",
            color="limegreen",
            fillcolor="white",
            fontcolor="limegreen",
            fontsize=str(NODE_FONTSIZE),
        )

        self.attr_nodes['sink'].update(
            shape="circle",
            rank="sink",
            peripheries="2",
            width="2.5",
            height="2.5",
            penwidth="4",
            color="orangered",
            fillcolor="white",
            fontcolor="orangered",
            fontsize=str(NODE_FONTSIZE),
        )

        self.attr_nodes['dependency'].update(
            width="3.0",
            height="1.8",
            penwidth="4.0",
            fontsize=str(NODE_FONTSIZE),
        )

        self.attr_nodes['bus'] = self.attr_nodes['default'].copy()
        self.attr_nodes['bus'].update(
            shape="rect",
            width="3.0",
            height="0.5",
            penwidth="4.0",
            labelloc="c",
            fontsize=str(NODE_FONTSIZE),
        )

        self.primary_node_types.append('bus')

        # ----------------------------------------------------------------------------
        # Apply node clustering, if defined in config

        if self.clustering and self.layout_engine not in ['neato', 'fdp', 'sfdp']:
            self.apply_node_clustering()

        # ----------------------------------------------------------------------------

        for node_id in list(self.component_attr.keys()):

            # Position assignment for nodes
            pos_x = self.component_attr[node_id]['pos_x']
            pos_y = self.component_attr[node_id]['pos_y']
            node_pos = ''
            if is_number(pos_x) and is_number(pos_y):
                node_pos = str(pos_x) + "," + str(pos_y) + "!"

            # Segment long labels to fit within nodes spaces
            label_mod = segment_long_labels(node_id, delims=['_', ' '], maxlen=12)

            # Custom attribute assignment based on node types
            node_type = str(self.component_attr[node_id]['node_type']).lower()

            if node_type not in self.primary_node_types:
                node_type = 'default'

            if node_type == 'default':
                self.agraph.get_node(node_id).attr.update(
                    label='', xlabel=label_mod, pos=node_pos,
                    **self.attr_nodes['default'])

            elif node_type in ['supply', 'source']:
                self.agraph.get_node(node_id).attr.update(
                    label=label_mod, pos=node_pos,
                    **self.attr_nodes['supply'])

            elif node_type in ['sink', 'consumer', 'output']:
                label_mod = segment_long_labels(node_id, delims=['_', ' '], maxlen=8)
                self.agraph.get_node(node_id).attr.update(
                    label=label_mod, pos=node_pos,
                    **self.attr_nodes['sink'])

            elif node_type in ['dependency']:
                label_mod = segment_long_labels(node_id, delims=['_', ' '], maxlen=8)
                self.agraph.get_node(node_id).attr.update(
                    label=label_mod, pos=node_pos,
                    **self.attr_nodes['dependency'])

            elif node_type == 'junction':
                self.agraph.get_node(node_id).attr.update(
                    label="", xlabel="", pos=node_pos,
                    **self.attr_nodes['junction'])

            else:
                self.agraph.get_node(node_id).attr.update(
                    label="", xlabel=label_mod, pos=node_pos,
                    **self.attr_nodes['default'])

            component_class = \
                str(self.component_attr[node_id]['component_class']).lower()
            if component_class == 'bus':
                label_mod = segment_long_labels(node_id, delims=['_', ' '], maxlen=7)
                label_mod = label_mod[1:-1]
                # xlabel_new = \
                #     "<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='0'>"\
                #     + "<TR><TD CELLPADDING='0' ALIGN='right'>"\
                #     + label_mod\
                #     + "</TD></TR></TABLE>>"
                xlabel_new = f"<<I>{label_mod}</I>>"

                self.agraph.get_node(node_id).attr.update(
                    label="", xlabel=xlabel_new,
                    **self.attr_nodes[component_class])

        # ----------------------------------------------------------------------------
        return self.agraph

# =====================================================================================


class SystemTopology_WTP(SystemTopology_Generic):

    def __init__(self, *args):
        super(SystemTopology_WTP, self).__init__(*args)
        self.graph_label = "Treatment Plant Topology"

    def draw_sys_topology(self):
        fname = Path(self.out_file).stem
        self.clustering = False
        self.build_graph_structure()
        self.configure_sys_topology()

        self.agraph.graph_attr.update(rankdir="TB", splines="spline")
        self.write_graphs_to_file(fname, dpi=300, engines=['dot'])

        self.agraph.graph_attr.update(rankdir="TB", splines="spline")
        self.write_graphs_to_file(fname, dpi=600, engines=['neato'])

    def configure_sys_topology(self):
        """
        Draws the component configuration for a given infrastructure system.

        :output: generates and saves the system topology diagram in the
        following formats: (graphviz) dot, png, pdf, ps2.
        """

        # ----------------------------------------------------------------------------
        self.agraph.graph_attr.update(
            orientation="portrait",
            concentrate="False",
            rankdir="TB",
            center="True",
            forcelabels="True",
            fontsize="60",
            pad="0.1",
            pack="False",
            sep="+1",
            overlap="False",
            smoothing="none",
            ratio="auto",
            scale="1",
        )

        self.agraph.edge_attr.update(
            arrowhead="normal",
            penwidth="50",
            # arrowsize="50",
            color=str(self.default_edge_color),
        )

        # ----------------------------------------------------------------------------
        # Customise node attributes based on type of infrastructure system

        NODE_FONTSIZE = 35

        self.attr_nodes['default'].update(
            shape="circle",
            style="filled",
            width="3.6",
            height="3.6",
            fixedsize="True",
            penwidth="3",
            fontsize=str(NODE_FONTSIZE),
        )

        self.attr_nodes['supply'].update(
            width="4.7",
            height="2.5",
            fixedsize="True",
            penwidth="3",
            fontsize=str(NODE_FONTSIZE),
        )

        self.attr_nodes['sink'].update(
            width="3.5",
            height="3.5",
            fixedsize="True",
            penwidth="3",
            fontsize=str(NODE_FONTSIZE),
        )

        self.attr_nodes['dependency'].update(
            width="4.7",
            height="2.5",
            fixedsize="True",
            penwidth="3",
            fontsize=str(NODE_FONTSIZE),
        )

        self.attr_nodes['large_basin'] = self.attr_nodes['default'].copy()
        self.attr_nodes['large_basin'].update(
            shape="rect", width="5", height="2.5")

        self.attr_nodes['small_basin'] = self.attr_nodes['default'].copy()
        self.attr_nodes['small_basin'].update(
            shape="rect", width="3.5", height="2")

        self.attr_nodes['chemical_tank'] = self.attr_nodes['default'].copy()
        self.attr_nodes['chemical_tank'].update(
            shape="circle", width="4.0", height="1.8", fixedsize="False")

        self.attr_nodes['building'] = self.attr_nodes['default'].copy()
        self.attr_nodes['building'].update(
            shape="box", style="rounded", width="4.5", height="2")

        self.attr_nodes['pump'] = self.attr_nodes['default'].copy()
        self.attr_nodes['pump'].update(
            shape="hexagon", width="4", height="4", fixedsize="False")

        self.attr_nodes['switchroom'] = self.attr_nodes['default'].copy()
        self.attr_nodes['switchroom'].update(
            shape="rect", style="rounded", width="3.2", height="1.8")

        self.primary_node_types.extend([
            'large_basin', 'small_basin', 'chemical_tank',
            'building', 'pump', 'switchroom'])

        # ----------------------------------------------------------------------------
        # Apply custom node attributes based on node type

        for node_id in list(self.component_attr.keys()):

            # Position assignment for nodes
            pos_x = self.component_attr[node_id]['pos_x']
            pos_y = self.component_attr[node_id]['pos_y']
            node_pos = ''
            if is_number(pos_x) and is_number(pos_y):
                node_pos = str(pos_x) + "," + str(pos_y) + "!"

            # Segment long labels to fit within nodes spaces
            label_mod = segment_long_labels(node_id, delims=['_', ' '], maxlen=10)

            # Custom attribute assignment based on node types
            node_type = str(self.component_attr[node_id]['node_type']).lower()

            component_class = \
                str(self.component_attr[node_id]['component_class']).lower()

            if node_type not in self.primary_node_types:
                node_type = 'default'

            if node_type in ['supply', 'source']:
                self.agraph.get_node(node_id).attr.update(
                    label=label_mod, pos=node_pos,
                    **self.attr_nodes['supply'])

            elif node_type in ['sink', 'consumer', 'output']:
                self.agraph.get_node(node_id).attr.update(
                    label=label_mod, pos=node_pos,
                    **self.attr_nodes['sink'])

            elif node_type in ['dependency']:
                self.agraph.get_node(node_id).attr.update(
                    label=label_mod, pos=node_pos,
                    **self.attr_nodes['dependency'])

            elif node_type == 'junction':
                self.agraph.get_node(node_id).attr.update(
                    label="", xlabel="", pos=node_pos,
                    **self.attr_nodes['junction'])

            else:
                self.agraph.get_node(node_id).attr.update(
                    label=label_mod, xlabel="",
                    pos=node_pos, **self.attr_nodes['default'])

            if component_class in [
                    'large tank', 'large basin', 'laarge_basin',
                    'sedimentation basin',
                    'sedimentation basin - large']:
                self.agraph.get_node(node_id).attr.update(
                    label=label_mod, xlabel="",
                    pos=node_pos, **self.attr_nodes['large_basin'])

            elif component_class in [
                    'small_basin', 'small tank', 'small basin',
                    'sedimentation basin - small',
                    'chlorination tank']:
                self.agraph.get_node(node_id).attr.update(
                    label=label_mod, xlabel="",
                    pos=node_pos, **self.attr_nodes['small_basin'])

            elif component_class in ['chemical tank', 'chemical_tank']:
                self.agraph.get_node(node_id).attr.update(
                    label=label_mod, xlabel="",
                    pos=node_pos, **self.attr_nodes['chemical_tank'])

            elif component_class in ['building', 'small building']:
                self.agraph.get_node(node_id).attr.update(
                    label=label_mod, xlabel="",
                    pos=node_pos, **self.attr_nodes['building'])

            elif component_class in ['pump', 'pumps']:
                self.agraph.get_node(node_id).attr.update(
                    label="", xlabel=label_mod,
                    pos=node_pos, **self.attr_nodes['pump'])

            elif component_class in ['switchroom', 'power supply']:
                self.agraph.get_node(node_id).attr.update(
                    label=label_mod, xlabel="",
                    pos=node_pos, **self.attr_nodes['switchroom'])

        # ----------------------------------------------------------------------------
        # Apply node clustering, if defined in config

        if self.clustering and self.layout_engine not in ['neato', 'fdp', 'sfdp']:
            self.apply_node_clustering()

        # ----------------------------------------------------------------------------
        return self.agraph

# =====================================================================================
