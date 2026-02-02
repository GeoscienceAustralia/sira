import igraph


class ComponentGraph(object):
    """
    Border class abstraction of the component graph in an attempt to optimise the
    calculation of economic loss by using different Graph packages.
    This implementation uses igraph
    """

    def __init__(self, components, comp_sample_func=None):
        """
        Construct a graph from the igraph package using the component dict.

        Parameters:
        -----------
        components:
            Dict of components that represent the infrastructure model
        comp_sample_func:
            Array of the functionality of each component (1.0 -> 0.0).
        """
        # create the directed graph
        self.digraph = igraph.Graph(directed=True)

        # if we don't have a functionality array create a default one with 1.0's
        if comp_sample_func is None:
            comp_sample_func = [1.0] * len(components)

        # Future improvement: consider making this part of components.  # noqa: W0511
        self.id_index_map = {
            comp_id: comp_index
            for comp_index, comp_id in list(enumerate(sorted(components.keys())))
        }

        # iterate through the components to create the graph
        # (using a list of sorted keys as we're trying to match the old code)
        for comp_id in components.keys():
            component = components[comp_id]
            comp_index = self.id_index_map[comp_id]
            # check that the component is not already in the graph
            if len(self.digraph.vs) == 0 or comp_id not in self.digraph.vs["name"]:  # noqa: E1136
                # add new component
                self.digraph.add_vertex(name=comp_id)

            # iterate through this components connected components
            for dest_comp_id, _ in component.destination_components.items():
                dest_index = self.id_index_map[dest_comp_id]
                # check if the parent component is a dependent node type
                if component.node_type == "dependency":
                    # combine the dependent nodes functionality
                    # Review the robustness of this logic.  # noqa: W0511
                    self.update_dependency(comp_sample_func, comp_index, dest_index)

                # Is the child node in the graph
                if len(self.digraph.vs) == 0 or dest_comp_id not in self.digraph.vs["name"]:  # noqa: E1136
                    # add new child component
                    self.digraph.add_vertex(name=dest_comp_id)

                # connect the parent and child vertices with an edge.
                # The functionality of the parent vertex is the value
                # of the edge capacity.
                self.digraph.add_edge(comp_id, dest_comp_id, capacity=comp_sample_func[comp_index])

        # Here digraph.vs is a `igraph.VertexSeq` object and
        # it can be used as an iterable
        for v in self.digraph.vs:  # noqa:E1133
            v["component_object"] = components[v["name"]]
        self.estimate_betweenness()

    def update_capacity(self, components, comp_sample_func):
        """
        Update the graph to change the edge's capacity value to
        reflect the new functionality of the parent vertice.
        """
        # iterate through the infrastructure components
        # (using a list of sorted keys as we're trying to match the old code)
        for comp_id in components.keys():
            component = components[comp_id]
            comp_index = self.id_index_map[comp_id]
            # iterate through the destination components
            for dest_comp_id in component.destination_components.keys():
                dest_index = self.id_index_map[dest_comp_id]

                if component.node_type == "dependency":
                    # combine the dependent vertices functionality with the parents
                    # *** VERIFY *** the logic of the following
                    self.update_dependency(comp_sample_func, comp_index, dest_index)

                # Determine the edge id from the id's of the
                # parent and child node
                edge_id = self.digraph.get_eid(comp_id, dest_comp_id)
                # update the edge dict with the functionality of the
                # parent vertex
                self.digraph.es[edge_id]["capacity"] = comp_sample_func[comp_index]  # noqa:E1136

    def update_dependency(self, comp_sample_func, parent, dependent):
        min_capacity = min(comp_sample_func[parent], comp_sample_func[dependent])
        comp_sample_func[dependent] = min_capacity

    def dump_graph(self, external=None):
        """
        Dump the contents of the graph.

        Logs at info level the edges of the graph, with the
        capacity value for each edge. Optionally will dump the passed
        graph, which must implement the igraph methods.
        """

        # Use the parameter graph if provided
        comp_graph = external if external else self.digraph
        # Iterate through the edges
        edge_dict = {}
        for edge in comp_graph.get_edgelist():
            # we need to use the two vertice id's to get the edge id
            edge_id = comp_graph.get_eid(edge[0], edge[1])
            # log the names of the vertices with the capacity of the edge
            edge_dict[edge_id] = {
                "vertices": (edge[0], edge[1]),
                "capacity": comp_graph.es[edge_id]["capacity"],
            }  # noqa:E1136
        return edge_dict

    def maxflow(self, supply_comp_id, output_comp_id):
        """Computes the maximum flow between two nodes."""

        # determine the vertex id's for the two components
        sup_v = self.digraph.vs.find(supply_comp_id)  # noqa:E1101
        out_v = self.digraph.vs.find(output_comp_id)  # noqa:E1101
        # calculate the maximum flow value between the two id's
        return self.digraph.maxflow_value(sup_v.index, out_v.index, self.digraph.es["capacity"])  # noqa:E1136

    def estimate_betweenness(self):
        centrality_estimate = self.digraph.betweenness()
        # Here digraph.vs is a `igraph.VertexSeq` object and
        # it can be used as an iterable
        for v in self.digraph.vs:  # noqa:E1133
            v["betweenness"] = centrality_estimate[v.index]
        # for v in self.digraph.vs:
        #     print("{:40} : {}".format(v["name"], v["betweenness"]))
