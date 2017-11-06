import unittest
import logging

import numpy as np
logging.basicConfig(level=logging.INFO)

from sifra.sysresponse import calc_loss_arrays, calc_sys_output, compute_output_given_ds
from sifra.sifraclasses import FacilitySystem, Scenario
from infrastructure_response import calculate_response, ingest_spreadsheet

config_file = '/opt/project/tests/test_scenario_ps_coal.conf'


class TestComponentGraph(unittest.TestCase):
    def test_graph_function(self):
        facility = FacilitySystem(config_file)
        infrastructure = ingest_spreadsheet(config_file)

        from random import random

        for _ in range(10000):
            function_level = random()
            component_idx = int(random()*31 + 1)
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

    def test_small_changes_in_function(self):
        # get two arrays of component function levels and create an array
        # of function values that make 100 steps between them. Iterate through
        # the component graph calculating the output
        facility = FacilitySystem(config_file)
        infrastructure = ingest_spreadsheet(config_file)

        # the following function levels were captured in a test run which had 15000 samples and was unseeded
        sr_func_array = np.array([[0.98799999999999999, 0.99580000000000002, 0.99453333333333338, 1.0, 0.99993333333333334,
              0.98999999999999999, 0.98746666666666671, 0.98806666666666665, 0.99946666666666661, 0.99973333333333336,
              1.0, 1.0, 0.64659999999999995, 0.64780000000000004, 0.64546666666666663, 0.65026666666666666,
              0.99119999999999997, 0.99993333333333334, 0.99993333333333334, 0.87313333333333332, 0.86933333333333329,
              0.86819999999999997, 0.98286666666666667, 0.98153333333333337, 0.9801333333333333, 0.98266666666666669,
              0.96226666666666671, 0.96273333333333333, 0.99919999999999998, 0.9993333333333333, 0.94017333333333319,
              0.94046666666666667, 0.99233333333333329],
                                  [0.90286666666666671, 0.9214, 0.92559999999999998, 1.0, 0.99726666666666663, 0.90013333333333334,
              0.8998666666666667, 0.89966666666666661, 0.98673333333333335, 0.98693333333333333, 1.0, 1.0,
              0.37133333333333335, 0.37213333333333332, 0.36533333333333334, 0.36533333333333334, 0.94899999999999995,
              0.99686666666666668, 0.99706666666666666, 0.74419999999999997, 0.74126666666666663, 0.7436666666666667,
              0.92766666666666664, 0.92833333333333334, 0.92866666666666664, 0.92186666666666661, 0.82386666666666664,
              0.82306666666666661, 0.98999999999999999, 0.98726666666666663, 0.86547999999999992, 0.86517333333333324,
              0.90406666666666669],
                                  [0.70679999999999998, 0.69486666666666663, 0.6986, 1.0, 0.98293333333333333, 0.68546666666666667,
              0.70479999999999998, 0.70199999999999996, 0.92466666666666664, 0.93006666666666671, 1.0, 1.0, 0.1852,
              0.18513333333333334, 0.18433333333333332, 0.18659999999999999, 0.86019999999999996, 0.97740000000000005,
              0.97453333333333336, 0.60999999999999999, 0.61339999999999995, 0.61653333333333338, 0.82140000000000002,
              0.83066666666666666, 0.82213333333333338, 0.82886666666666664, 0.60819999999999996, 0.60333333333333339,
              0.92679999999999996, 0.9244, 0.77735999999999994, 0.77597333333333318, 0.68820000000000003]])

        if_func_array = np.array([[0.98919999999999997, 0.99480000000000002, 0.99560000000000004, 1.0, 0.99993333333333334,
                  0.98980000000000001, 0.98740000000000006, 0.98926666666666663, 0.9993333333333333,
                  0.99946666666666661, 1.0, 1.0, 0.65673333333333328, 0.65493333333333337, 0.64980000000000004,
                  0.65293333333333337, 0.99039999999999995, 1.0, 0.99993333333333334, 0.87453333333333338,
                  0.86699999999999999, 0.86706666666666665, 0.9821333333333333, 0.98033333333333328,
                  0.97873333333333334, 0.98199999999999998, 0.96326666666666672, 0.96299999999999997,
                  0.99926666666666664, 0.99973333333333336, 0.94140000000000001, 0.94274666666666673,
                  0.99080000000000001],
                                  [0.9045333333333333, 0.92020000000000002, 0.92113333333333336, 1.0, 0.99726666666666663,
              0.89459999999999995, 0.90326666666666666, 0.90033333333333332, 0.98740000000000006, 0.98773333333333335,
              1.0, 1.0, 0.37226666666666669, 0.37319999999999998, 0.36699999999999999, 0.37853333333333333,
              0.94853333333333334, 0.99733333333333329, 0.99753333333333338, 0.73986666666666667, 0.74486666666666668,
              0.7436666666666667, 0.9257333333333333, 0.9250666666666667, 0.9274, 0.92520000000000002,
              0.82406666666666661, 0.82293333333333329, 0.98660000000000003, 0.98719999999999997, 0.86918666666666666,
              0.8639066666666666, 0.90300000000000002],
                                  [0.71146666666666669, 0.70253333333333334, 0.6996, 1.0, 0.98360000000000003, 0.67859999999999998,
              0.70473333333333332, 0.70586666666666664, 0.92766666666666664, 0.92546666666666666, 1.0, 1.0,
              0.18693333333333334, 0.18646666666666667, 0.1842, 0.18533333333333332, 0.85606666666666664,
              0.97453333333333336, 0.97560000000000002, 0.60860000000000003, 0.61099999999999999, 0.61060000000000003,
              0.82746666666666668, 0.82766666666666666, 0.82726666666666671, 0.8266, 0.61099999999999999,
              0.60999999999999999, 0.92679999999999996, 0.9250666666666667, 0.77493333333333336, 0.77171999999999996,
              0.68786666666666663]])

        # loop through both component lists for each pga 0.3 to 0.5
        for pga_level in range(0, 3):
            print("Printing means for {}".format(pga_level))
            sr_comp_func_level = sr_func_array[pga_level]
            print("sr level mean {}".format(np.mean(sr_comp_func_level)))
            if_comp_func_level = if_func_array[pga_level]
            print("if level mean {}".format(np.mean(if_comp_func_level)))
            print("Mean % difference between sr and if {}".format(np.mean((sr_comp_func_level-if_comp_func_level)/sr_comp_func_level)))

            # calculate if output
            if_output = infrastructure.compute_output_given_ds(if_comp_func_level)
            infrastructure.component_graph.dump_graph()
            print("if output of if {}".format(if_output))
            srif_output = compute_output_given_ds(if_comp_func_level, facility)
            if np.any(if_output != srif_output):
                logging.info("Dumping sr graph")
                infrastructure.component_graph.dump_graph(facility.network.G)

            # calculated sr output
            sr_output = compute_output_given_ds(sr_comp_func_level, facility)
            print("sr output of sr {}".format(sr_output))
            infrastructure.component_graph.dump_graph(facility.network.G)
            ifsr_output = infrastructure.compute_output_given_ds(sr_comp_func_level)
            if np.any(sr_output != ifsr_output):
                logging.info("Dumping if graph")
                infrastructure.component_graph.dump_graph()

        logging.info("test complete")
