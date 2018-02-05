import unittest
import logging

import numpy as np
logging.basicConfig(level=logging.INFO)

from sifra.sysresponse import calc_loss_arrays, calc_sys_output, compute_output_given_ds
from sifra.sifraclasses import FacilitySystem, Scenario
from sifra.infrastructure_response import calculate_response, ingest_spreadsheet

config_file = './test_scenario_ps_coal.conf'
series_config = './test_simple_series_struct.conf'
parallel_config = './test_simple_parallel_struct.conf'
dependent_config = './test_simple_series_struct_dep.conf'


class TestComponentGraph(unittest.TestCase):
    def test_graph_function(self):
        facility = FacilitySystem(config_file)
        infrastructure, _ = ingest_spreadsheet(config_file)

        # seeding is not used
        prng = np.random.RandomState()

        for rand_func, rand_comp in prng.uniform(size=(10000, 2)):
            function_level = rand_func
            component_idx = int(rand_comp*31 + 1)
            function_list = [1.0]*33
            function_list[component_idx] = function_level
            sys_output = compute_output_given_ds(function_list, facility)

            function_list = [1.0]*33
            function_list[component_idx] = function_level
            if_output = infrastructure.compute_output_given_ds(function_list)

            if sys_output[0] != if_output[0] or \
                            sys_output[1] != if_output[1]:
                print("\n{} {} sr:{} if:{}".format(function_level,
                                                 component_idx,
                                                 sys_output,
                                                 if_output))
                # dump trees
                print("\nsys graph")
                infrastructure._component_graph.dump_graph(facility.network.G)
                print("\nif graph")
                infrastructure._component_graph.dump_graph()

    def test_source_break(self):
        facility = FacilitySystem(config_file)

        infrastructure, _ = ingest_spreadsheet(config_file)

        functionality = 0.316189922718
        comp_idx = 5
        function_list = [1.0] * 33
        function_list[comp_idx] = functionality
        # sr:[300.0, 300.0] if:[189.71395363  189.71395363]
        sys_output = compute_output_given_ds(function_list, facility)

        # reset the functionality array
        function_list = [1.0] * 33
        function_list[comp_idx] = functionality
        if_output = infrastructure.compute_output_given_ds(function_list)

        if sys_output[0] != if_output[0] or \
                        sys_output[1] != if_output[1]:
            print("\n5 0.31 sr:{} if:{}".format(sys_output,
                                                if_output))
            # dump trees
            print("\nsys graph")
            infrastructure._component_graph.dump_graph(facility.network.G)
            print("\nif graph")
            infrastructure._component_graph.dump_graph()

    def test_duplicate_break(self):
        facility = FacilitySystem(config_file)

        infrastructure, _ = ingest_spreadsheet(config_file)

        functionality = 0.455400408676
        comp_idx = 16
        function_list = [1.0] * 33
        function_list[comp_idx] = functionality

        # sr:[300.0, 300.0] if:[189.71395363  189.71395363]
        sys_output = compute_output_given_ds(function_list, facility)

        # reset the functionality array
        function_list = [1.0] * 33
        function_list[comp_idx] = functionality
        if_output = infrastructure.compute_output_given_ds(function_list)

        if sys_output[0] != if_output[0] or \
                        sys_output[1] != if_output[1]:
            print("\n5 0.31 sr:{} if:{}".format(sys_output,
                                                if_output))
            # dump trees
            print("\nsys graph")
            infrastructure._component_graph.dump_graph(facility.network.G)
            print("\nif graph")
            infrastructure._component_graph.dump_graph()

    def test_func_changes_in_series(self):
        series_if, _ = ingest_spreadsheet(series_config)
        function_array = np.ones(len(series_if.components))
        pga_range = np.arange(1.0, 0.0, -0.001)
        results = np.zeros(pga_range.shape)
        logging.info("series n1")
        for index, test_pga in enumerate(pga_range):
            function_array[1] = test_pga
            max_flow = series_if.compute_output_given_ds(function_array)
            results[index] = max_flow

        self.dump_stats(pga_range, results, "series n1")
        series_if._component_graph.dump_graph()
        function_array = np.ones(len(series_if.components))
        results_2 = np.zeros(pga_range.shape)
        logging.info("series n2")
        for index, test_pga in enumerate(pga_range):
            function_array[2] = test_pga
            max_flow = series_if.compute_output_given_ds(function_array)
            results_2[index] = max_flow

        self.dump_stats(pga_range, results, "series n2")
        series_if._component_graph.dump_graph()
        logging.info("test complete")

    def test_func_changes_in_parallel(self):
        logging.info("\n")
        series_if, _ = ingest_spreadsheet(parallel_config)
        pga_range = np.arange(1.0, 0.0, -0.001)

        for node_number in [1, 2, 3]:
            results = np.zeros(pga_range.shape)
            function_array = np.ones(len(series_if.components))
            logging.info("parallel n{}".format(node_number))
            for index, test_pga in enumerate(pga_range):
                function_array[node_number] = test_pga
                max_flow = series_if.compute_output_given_ds(function_array)
                results[index] = max_flow

            self.dump_stats(pga_range, results, "parallel_n{}".format(node_number))
            series_if._component_graph.dump_graph()

        logging.info("test complete")

    def test_ds_iteration_in_para(self):
        logging.info("\n")
        series_if, _ = ingest_spreadsheet(parallel_config)
        ds_range = np.arange(0, 5)

        for node_number in [1, 2, 3]:
            results = np.zeros(ds_range.shape)
            function_array = np.ones(len(series_if.components))
            logging.info("parallel n{}".format(node_number))
            for index, test_ds in enumerate(ds_range):
                component = series_if.components.index(node_number)
                function_array[node_number] = component.frag_func.damage_states.index(test_ds).functionality
                max_flow = series_if.compute_output_given_ds(function_array)
                results[index] = max_flow

            self.dump_stats(ds_range,
                            results,
                            "ds_para n{}".format(node_number),
                            "damage state")
            series_if._component_graph.dump_graph()

        logging.info("test complete")

    def test_func_changes_in_dep(self):
        logging.info("\n")
        series_if, _ = ingest_spreadsheet(dependent_config)
        pga_range = np.arange(1.0, 0.0, -0.001)

        for node_number in [1, 4, 3]:
            results = np.zeros(pga_range.shape)
            function_array = np.ones(len(series_if.components))
            logging.info("dep_n{}".format(node_number))
            for index, test_pga in enumerate(pga_range):
                function_array[node_number] = test_pga
                max_flow = series_if.compute_output_given_ds(function_array)
                results[index] = max_flow

            self.dump_stats(pga_range, results, "dep_n{}".format(node_number))
            series_if._component_graph.dump_graph()

        logging.info("test complete")

    def test_ds_iteration_in_dep(self):
        logging.info("\n")
        series_if, _ = ingest_spreadsheet(dependent_config)
        ds_range = np.arange(0, 5)

        for node_number in [1, 4, 3]:
            results = np.zeros(ds_range.shape)
            function_array = np.ones(len(series_if.components))
            logging.info("dep_n{}".format(node_number))
            for index, test_ds in enumerate(ds_range):
                component = series_if.components.index(node_number)
                function_array[node_number] = component.frag_func.damage_states.index(test_ds).functionality
                max_flow = series_if.compute_output_given_ds(function_array)
                results[index] = max_flow

            self.dump_stats(ds_range,
                            results,
                            "ds_dep_n{}".format(node_number),
                            "damage state")
            series_if._component_graph.dump_graph()

        logging.info("test complete")

    def dump_stats(self, pga_range, results_array, title, x_label="functionality %"):
        import matplotlib.pyplot as plt

        logging.info("mean {}".format(np.mean(results_array)))
        logging.info("std {}".format(np.std(results_array)))
        hist, bin = np.histogram(results_array)
        logging.info("hist {}".format(hist))
        logging.info("bin {}".format(bin))
        fig, ax = plt.subplots()
        ax.axis([pga_range[0], pga_range[-1], 0, max(results_array)+10])
        ax.scatter(pga_range, results_array)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("flow")
        fig.savefig(title+'.png', format='png', bbox_inches='tight', dpi=300)
