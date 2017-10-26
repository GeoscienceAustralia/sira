import igraph
import numpy as np
import logging


class ComponentGraph(object):
    """
    Border class abstraction of the component graph in an attempt to optimise the
    calculation of economic loss by using different Graph packages.
    This implementation uses igraph
    """
    def __init__(self, components, comp_sample_func):
        self.comp_graph = igraph.Graph(directed=True)
        self.dumped = False

        for comp_index, comp_id in enumerate(sorted(components.keys())):
            component = components[comp_id]
            if len(self.comp_graph.vs) == 0 or comp_id not in self.comp_graph.vs['name']:
                self.comp_graph.add_vertex(name=comp_id)

            for dest_index, (dest_comp_id, destination_component) in enumerate(
                component.destination_components.items()):

                if component.node_type == 'dependency':
                    comp_sample_func[dest_index] *= comp_sample_func[comp_index]

                if len(self.comp_graph.vs) == 0 or dest_comp_id not in self.comp_graph.vs['name']:
                    self.comp_graph.add_vertex(name=dest_comp_id)

                self.comp_graph.add_edge(comp_id, dest_comp_id,
                                         capacity=comp_sample_func[comp_index])

    def update_capacity(self, components, comp_sample_func):
        for comp_index, comp_id in enumerate(sorted(components.keys())):
            component = components[comp_id]
            for dest_index, dest_comp_id in enumerate(component.destination_components.keys()):
                if component.node_type == 'dependency':
                    comp_sample_func[dest_index] *= comp_sample_func[comp_index]

                edge_id = self.comp_graph.get_eid(comp_id, dest_comp_id)
                self.comp_graph.es[edge_id]['capacity'] = comp_sample_func[comp_index]

        if 0.5 > np.min(comp_sample_func) and not self.dumped:
            self.dumped = True
            logging.info("\nif_resp func graph mean {}".format(np.mean(comp_sample_func)))
            # self.dump_graph()

    def dump_graph(self, external=None):
        comp_graph = external if external else self.comp_graph
        for edge in comp_graph.get_edgelist():
            edge_id = comp_graph.get_eid(edge[0], edge[1])
            logging.info("{}->{} = {}".format(comp_graph.vs[edge[0]]['name'],
                                           comp_graph.vs[edge[1]]['name'],
                                           comp_graph.es[edge_id]['capacity']))

    def maxflow(self, supply_comp_id, output_comp_id):
        sup_v = self.comp_graph.vs.find(supply_comp_id)
        out_v = self.comp_graph.vs.find(output_comp_id)

        return self.comp_graph.maxflow_value(sup_v.index,
                                             out_v.index,
                                             self.comp_graph.es['capacity'])
