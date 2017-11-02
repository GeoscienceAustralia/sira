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

        component_count = len(infrastructure.components)
        step_count = 1000
        # the following function levels were captured in a test run which had 15000 samples and was unseeded
        if_func_array = np.array([[0.96353333333333335, 0.92286666666666661, 0.99973333333333336, 0.99993333333333334, 1.0, 0.91926666666666668,
             1.0, 0.91873333333333329, 0.99806666666666666, 1.0, 0.98561333333333323, 0.99993333333333334, 1.0, 1.0,
             0.96279999999999999, 0.96179999999999999, 0.99839999999999995, 0.99853333333333338, 0.99919999999999998, 1.0,
             0.99986666666666668, 0.99993333333333334, 0.99986666666666668, 1.0, 1.0, 1.0, 1.0, 0.98531999999999997,
             0.99880000000000002, 0.99846666666666661, 0.99993333333333334, 0.99993333333333334, 0.9194],
         [0.87419999999999998, 0.65106666666666668, 0.99039999999999995, 0.98933333333333329, 1.0, 0.64626666666666666,
             1.0, 0.64766666666666661, 0.96099999999999997, 0.9993333333333333, 0.94281333333333328, 0.99113333333333331,
             1.0, 1.0, 0.87233333333333329, 0.86880000000000002, 0.98433333333333328, 0.98219999999999996,
             0.98166666666666669, 1.0, 0.98880000000000001, 0.98719999999999997, 0.98733333333333329, 0.9993333333333333,
             0.9993333333333333, 1.0, 0.9996666666666667, 0.94209333333333334, 0.98326666666666662, 0.96240000000000003,
             0.99453333333333338, 0.99526666666666663, 0.6448666666666667],
         [0.73993333333333333, 0.36899999999999999, 0.9489333333333333, 0.90406666666666669, 1.0, 0.36899999999999999,
             1.0, 0.37653333333333333, 0.82646666666666668, 0.98833333333333329, 0.86963999999999986, 0.90946666666666665,
             0.9976666666666667, 0.99693333333333334, 0.73653333333333337, 0.73793333333333333, 0.92493333333333339,
             0.92320000000000002, 0.92633333333333334, 0.99660000000000004, 0.89693333333333336, 0.90533333333333332,
             0.89993333333333336, 0.98766666666666669, 0.98833333333333329, 1.0, 0.98793333333333333, 0.86895999999999984,
             0.9240666666666667, 0.82520000000000004, 0.92400000000000004, 0.92073333333333329, 0.37166666666666665],
         [0.6071333333333333, 0.18173333333333333, 0.85599999999999998, 0.70279999999999998, 1.0, 0.18566666666666667,
             1.0, 0.18206666666666665, 0.60966666666666669, 0.92700000000000005, 0.7719866666666666, 0.68579999999999997,
             0.97519999999999996, 0.9771333333333333, 0.61366666666666669, 0.61299999999999999, 0.83206666666666662,
             0.82040000000000002, 0.82286666666666664, 0.98280000000000001, 0.67533333333333334, 0.70186666666666664,
             0.70520000000000005, 0.92859999999999998, 0.92559999999999998, 1.0, 0.9254, 0.77794666666666656,
             0.8232666666666667, 0.61126666666666662, 0.70046666666666668, 0.69986666666666664, 0.18533333333333332]])

        sr_func_array = np.array([[0.99993333333333334, 0.99993333333333334, 0.99986666666666668, 1.0, 1.0, 1.0, 0.99973333333333336,
             0.99993333333333334, 1.0, 1.0, 1.0, 1.0, 0.91439999999999999, 0.9204, 0.91826666666666668, 0.91633333333333333,
             1.0, 1.0, 1.0, 0.96246666666666669, 0.96366666666666667, 0.96173333333333333, 0.99826666666666664,
             0.99919999999999998, 0.99880000000000002, 0.99853333333333338, 0.999, 0.99860000000000004, 1.0, 1.0,
             0.98582666666666663, 0.98528000000000004, 1.0],
        [0.98819999999999997, 0.99406666666666665, 0.99513333333333331, 1.0, 1.0, 0.98826666666666663,
             0.98826666666666663, 0.98833333333333329, 0.99926666666666664, 0.99906666666666666, 1.0, 1.0,
             0.64813333333333334, 0.65059999999999996, 0.65313333333333334, 0.64773333333333338, 0.99019999999999997, 1.0,
             1.0, 0.86726666666666663, 0.86613333333333331, 0.86613333333333331, 0.98346666666666671, 0.97953333333333337,
             0.98160000000000003, 0.98253333333333337, 0.96299999999999997, 0.96299999999999997, 0.99960000000000004,
             0.99939999999999996, 0.94075999999999993, 0.94268000000000007, 0.99119999999999997],
        [0.90473333333333328, 0.9237333333333333, 0.92346666666666666, 1.0, 0.99739999999999995, 0.89646666666666663,
             0.90200000000000002, 0.90000000000000002, 0.98713333333333331, 0.98793333333333333, 1.0, 1.0,
             0.36759999999999998, 0.36759999999999998, 0.36346666666666666, 0.36853333333333332, 0.9512666666666667,
             0.99693333333333334, 0.99680000000000002, 0.74526666666666663, 0.74506666666666665, 0.74226666666666663,
             0.92173333333333329, 0.92659999999999998, 0.92333333333333334, 0.92420000000000002, 0.82120000000000004,
             0.82479999999999998, 0.98633333333333328, 0.98740000000000006, 0.86803999999999992, 0.86756,
             0.90133333333333332],
        [0.70906666666666662, 0.70046666666666668, 0.69486666666666663, 1.0, 0.98153333333333337, 0.68386666666666662,
             0.70179999999999998, 0.69653333333333334, 0.92953333333333332, 0.92806666666666671, 1.0, 1.0,
             0.18513333333333334, 0.18726666666666666, 0.18326666666666666, 0.18733333333333332, 0.85933333333333328,
             0.97466666666666668, 0.97533333333333339, 0.61093333333333333, 0.60899999999999999, 0.61199999999999999,
             0.82499999999999996, 0.82753333333333334, 0.82633333333333336, 0.82333333333333336, 0.60880000000000001,
             0.60706666666666664, 0.92659999999999998, 0.9307333333333333, 0.77358666666666664, 0.77434666666666663,
             0.68753333333333333]])

        # loop through both component lists for each pga 0.3 to 0.5
        for pga_level in range(0, 4):
            print("Printing means for {}".format(pga_level))
            sr_comp_func_level = sr_func_array[pga_level]
            print("sr level mean {}".format(np.mean(sr_comp_func_level)))
            if_comp_func_level = if_func_array[pga_level]
            print("if level mean {}".format(np.mean(if_comp_func_level)))

            # calculated sr output
            sr_output = compute_output_given_ds(sr_comp_func_level, facility)
            print("sr output of sr {}".format(sr_output))
            ifsr_output=infrastructure.compute_output_given_ds(sr_comp_func_level)
            print("if output of sr input {}".format(ifsr_output))
            if np.any(sr_output != ifsr_output):
                infrastructure.component_graph.dump_graph()
                infrastructure.component_graph.dump_graph(facility.network.G)

            srif_output = compute_output_given_ds(if_comp_func_level, facility)
            print("sr output of if input {}".format(srif_output))
            # calculate if output
            if_output = infrastructure.compute_output_given_ds(if_comp_func_level)
            print("if output of if {}".format(if_output))
            if np.any(if_output != srif_output):
                logging.info("Dumping if graph")
                infrastructure.component_graph.dump_graph()
                logging.info("Dumping sr graph")
                infrastructure.component_graph.dump_graph(facility.network.G)


