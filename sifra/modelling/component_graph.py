import igraph


class ComponentGraph():
    """
    Abstraction of component graph in an attempt to optimise the calculation
    of economic loss
    of economic loss
    """
    def __init__(self, components, comp_sample_func):
        self.test_graph = igraph.Graph(directed=True)

        for comp_index, (comp_id, component) in enumerate(components.iteritems()):
            self.test_graph.add_vertex(name=comp_id)
            for dest_index, (dest_comp_id, destination_component) in enumerate(
                component.destination_components.iteritems()):

                if component.node_type == 'dependency':
                    comp_sample_func[dest_index] *= comp_sample_func[comp_index]

            self.test_graph.add_vertex(name=dest_comp_id)

            self.test_graph.add_edge(comp_id, dest_comp_id,
                                     capacity=comp_sample_func[comp_index],
                                     weight=component.weight)

    def maxflow(self, output_nodes, supply_nodes):
        system_flows_sample = []
        system_outflows_sample = np.zeros(len(self.output_nodes))
        for output_index, (output_comp_id, output_comp) in enumerate(self.output_nodes.iteritems()):
            # track the outputs by source type
            total_supply_flow_by_source = {}
            for supply_index, (supply_comp_id, supply_comp) in enumerate(self.supply_nodes.iteritems()):
                if_flow_fraction = self.test_graph.maxflow_value(supply_index,output_index)
                if_sample_flow = if_flow_fraction * supply_comp['capacity_fraction']

                if supply_comp['commodity_type'] not in total_supply_flow_by_source:
                    total_supply_flow_by_source[supply_comp['commodity_type']] = if_sample_flow
                else:
                    total_supply_flow_by_source[supply_comp['commodity_type']] += if_sample_flow

                system_flows_sample.append(tuple([supply_comp['commodity_type'],
                                                  supply_comp_id,
                                                  output_comp_id,
                                                  if_sample_flow]))

            total_available_flow = min(total_supply_flow_by_source.itervalues())

            estimated_capacity_fraction = min(total_available_flow, output_comp['capacity_fraction'])
            system_outflows_sample[output_index] = estimated_capacity_fraction * self.if_nominal_output

        return system_outflows_sample

