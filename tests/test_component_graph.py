import unittest
import logging

import numpy as np
logging.basicConfig(level=logging.INFO)

from sifra.sysresponse import calc_loss_arrays, calc_sys_output, compute_output_given_ds
from sifra.sifraclasses import FacilitySystem, Scenario
from infrastructure_response import calculate_response, ingest_spreadsheet

config_file = '/opt/project/tests/test_scenario_ps_coal.conf'
series_config = '/opt/project/tests/test_simple_series_struct.conf'
parallel_config = '/opt/project/tests/test_simple_parallel_struct.conf'
dependent_config = '/opt/project/tests/test_simple_series_struct_dep.conf'


class TestComponentGraph(unittest.TestCase):
    def test_graph_function(self):
        facility = FacilitySystem(config_file)
        infrastructure = ingest_spreadsheet(config_file)

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
                infrastructure.component_graph.dump_graph(facility.network.G)
                print("\nif graph")
                infrastructure.component_graph.dump_graph()

    def test_graph_function_much_damage(self):
        facility = FacilitySystem(config_file)
        infrastructure = ingest_spreadsheet(config_file)

        # seeding is not used
        prng = np.random.RandomState()

        for function_list in prng.uniform(size=(100000, 33)):
            sys_output = compute_output_given_ds(function_list.copy(), facility)
            if_output = infrastructure.compute_output_given_ds(function_list)

            if sys_output[0] != if_output[0] or \
                            sys_output[1] != if_output[1]:
                logging.info("\n{} \n sr:{} if:{}".format(function_list,
                                                   sys_output,
                                                   if_output))
                # dump trees
                logging.info("\nsys graph")
                infrastructure.component_graph.dump_graph(facility.network.G)
                logging.info("\nif graph")
                infrastructure.component_graph.dump_graph()
                self.fail("Massive!")

    def test_graph_dependency_nodes(self):
        breaker_array = np.array([0.22703707, 0.25474665, 0.15173351, 0.30446426, 0.03286735,
                                  0.48631226, 0.42523123, 0.53783504, 0.67832764, 0.50216505,
                                  0.4573121, 0.68553109, 0.57631243, 0.88695529, 0.39632882,
                                  0.13494193, 0.85481656, 0.02530913, 0.01912627, 0.59846684,
                                  0.3169484, 0.60619278, 0.73805362, 0.83371636, 0.32431238,
                                  0.72273922, 0.82481816, 0.53597114, 0.85886813, 0.23147034,
                                  0.35489199, 0.28757192, 0.73853101])
        facility = FacilitySystem(config_file)
        infrastructure = ingest_spreadsheet(config_file)

        sys_output = compute_output_given_ds(breaker_array.copy(), facility)
        if_output = infrastructure.compute_output_given_ds(breaker_array)

        self.assertTrue(sys_output[0] == if_output[0])
        self.assertTrue(sys_output[1] == if_output[1])

    def test_source_break(self):
        facility = FacilitySystem(config_file)

        infrastructure = ingest_spreadsheet(config_file)

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
            infrastructure.component_graph.dump_graph(facility.network.G)
            print("\nif graph")
            infrastructure.component_graph.dump_graph()

    def test_duplicate_break(self):
        facility = FacilitySystem(config_file)

        infrastructure = ingest_spreadsheet(config_file)

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
            infrastructure.component_graph.dump_graph(facility.network.G)
            print("\nif graph")
            infrastructure.component_graph.dump_graph()

    def test_func_changes_in_series(self):
        series_if = ingest_spreadsheet(series_config)
        function_array = np.ones(len(series_if.components))
        pga_range = np.arange(1.0, 0.0, -0.001)
        results = np.zeros(pga_range.shape)
        logging.info("series n1")
        for index, test_pga in enumerate(pga_range):
            function_array[1] = test_pga
            max_flow = series_if.compute_output_given_ds(function_array)
            results[index] = max_flow

        self.dump_stats(pga_range, results, "series n1")
        series_if.component_graph.dump_graph()
        function_array = np.ones(len(series_if.components))
        results_2 = np.zeros(pga_range.shape)
        logging.info("series n2")
        for index, test_pga in enumerate(pga_range):
            function_array[2] = test_pga
            max_flow = series_if.compute_output_given_ds(function_array)
            results_2[index] = max_flow

        self.dump_stats(pga_range, results, "series n2")
        series_if.component_graph.dump_graph()
        logging.info("test complete")

    def test_func_changes_in_parallel(self):
        logging.info("\n")
        series_if = ingest_spreadsheet(parallel_config)
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
            series_if.component_graph.dump_graph()

        logging.info("test complete")

    def test_ds_iteration_in_para(self):
        logging.info("\n")
        series_if = ingest_spreadsheet(parallel_config)
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
            series_if.component_graph.dump_graph()

        logging.info("test complete")

    def test_func_changes_in_dep(self):
        logging.info("\n")
        series_if = ingest_spreadsheet(dependent_config)
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
            series_if.component_graph.dump_graph()

        logging.info("test complete")

    def test_ds_iteration_in_dep(self):
        logging.info("\n")
        series_if = ingest_spreadsheet(dependent_config)
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
            series_if.component_graph.dump_graph()

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
