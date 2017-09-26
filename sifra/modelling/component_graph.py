import igraph
import numpy as np


class ComponentGraph(object):
    """
    Abstraction of component graph in an attempt to optimise the calculation
    of economic loss
    of economic loss
    """
    def __init__(self, components, comp_sample_func):
        self.comp_graph = igraph.Graph(directed=True)

        for comp_index, (comp_id, component) in enumerate(components.items()):
            self.comp_graph.add_vertex(name=comp_id)

            for dest_index, (dest_comp_id, destination_component) in enumerate(
                component.destination_components.items()):

                if component.node_type == 'dependency':
                    comp_sample_func[dest_index] *= comp_sample_func[comp_index]

                self.comp_graph.add_vertex(name=dest_comp_id)

                self.comp_graph.add_edge(comp_id, dest_comp_id,
                                         capacity=comp_sample_func[comp_index])

    def update_capacity(self, components, comp_sample_func):
        for comp_index, (comp_id, component) in enumerate(components.items()):
            for dest_index, dest_comp_id in enumerate(component.destination_components.keys()):
                if component.node_type == 'dependency':
                    comp_sample_func[dest_index] *= comp_sample_func[comp_index]

                edge_id = self.comp_graph.get_eid(comp_id, dest_comp_id)
                self.comp_graph.es[edge_id]['capacity'] = comp_sample_func[comp_index]

    def maxflow(self, supply_comp_id,output_comp_id):
        sup_v = self.comp_graph.vs.find(supply_comp_id)
        out_v = self.comp_graph.vs.find(output_comp_id)

        return self.comp_graph.maxflow_value(sup_v.index,
                                             out_v.index,
                                             self.comp_graph.es['capacity'])
