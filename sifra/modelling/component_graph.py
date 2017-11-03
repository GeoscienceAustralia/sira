import logging

import igraph


class ComponentGraph(object):
    """
    Border class abstraction of the component graph in an attempt to optimise the
    calculation of economic loss by using different Graph packages.
    This implementation uses igraph
    """
    def __init__(self, components, comp_sample_func):
        """
        Construct a graph from the igraph package using the component dict.
        :param components: Dict of components that represent the infrastructure model
        :param comp_sample_func: Array of the functionality of each component (1.0 -> 0.0).
        """
        # create the directed graph
        self.comp_graph = igraph.Graph(directed=True)
        self.dumped = False
        # store a map that will convert 'stack_1' -> 17 for editing the functionality (comp_sample_func)
        id_index_map = {v: k for k, v in list(enumerate(sorted(components.keys())))}

        # iterate through the components to create the graph
        for comp_index, comp_id in enumerate(sorted(components.keys())):
            component = components[comp_id]
            # check that the component is not already in the graph
            if len(self.comp_graph.vs) == 0 or comp_id not in self.comp_graph.vs['name']:
                # add new component
                self.comp_graph.add_vertex(name=comp_id)

            # iterate through this components connected components
            for dest_comp_id, destination_component in component.destination_components.items():
                dest_index = id_index_map[dest_comp_id]
                # check if the parent component is a dependent node type
                if component.node_type == 'dependency':
                    # combine the dependent nodes functionality
                    comp_sample_func[dest_index] *= comp_sample_func[comp_index]

                # Is the child node in the graph
                if len(self.comp_graph.vs) == 0 or dest_comp_id not in self.comp_graph.vs['name']:
                    # add new child component
                    self.comp_graph.add_vertex(name=dest_comp_id)

                # connect the parent and child vertices with an edge.
                # The functionality of the parent node is the
                self.comp_graph.add_edge(comp_id, dest_comp_id,
                                         capacity=comp_sample_func[comp_index])

    def update_capacity(self, components, comp_sample_func):
        """Update the graph to change any edge's capacity value to
        reflect the new functionality of the vertice."""
        id_index_map = {v: k for k, v in list(enumerate(sorted(components.keys())))}
        for comp_index, comp_id in enumerate(sorted(components.keys())):
            component = components[comp_id]
            for dest_comp_id in component.destination_components.keys():
                dest_index = id_index_map[dest_comp_id]
                if component.node_type == 'dependency':
                    comp_sample_func[dest_index] *= comp_sample_func[comp_index]

                edge_id = self.comp_graph.get_eid(comp_id, dest_comp_id)
                self.comp_graph.es[edge_id]['capacity'] = comp_sample_func[comp_index]

    def dump_graph(self, external=None):
        """
        Dump the contents of the graph.

        Logs at info level the edges of the graph, with the
        capacity value for each edge. Optionally will dump the passed
        graph, which must implement the igraph methods."""
        comp_graph = external if external else self.comp_graph
        for edge in comp_graph.get_edgelist():
            edge_id = comp_graph.get_eid(edge[0], edge[1])
            logging.info("{}->{} = {}".format(comp_graph.vs[edge[0]]['name'],
                                           comp_graph.vs[edge[1]]['name'],
                                           comp_graph.es[edge_id]['capacity']))

    def maxflow(self, supply_comp_id, output_comp_id):
        """Computes the maximum flow between two nodes."""
        sup_v = self.comp_graph.vs.find(supply_comp_id)
        out_v = self.comp_graph.vs.find(output_comp_id)

        return self.comp_graph.maxflow_value(sup_v.index,
                                             out_v.index,
                                             self.comp_graph.es['capacity'])
