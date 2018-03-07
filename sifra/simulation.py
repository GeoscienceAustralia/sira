from sifra.logger import rootLogger
from sifra.configuration import Configuration
from sifra.scenario import Scenario
from sifra.model_ingest import ingest_model
from sifra.modelling.system_topology import SystemTopology
from sifra.infrastructure_response import calculate_response, post_processing


def run_scenario(configuration_file_path):
    """
    Run a scenario by constructing a facility, and executing a scenario, with
    the parameters read from the config file.
    :param configuration_file_path: Scenario setting values and the infrastructure configuration file path
    :return: None
    """
    # Construct the scenario object
    rootLogger.info("Loading scenario config... ")

    config = Configuration(configuration_file_path)

    scenario = Scenario(config)
    rootLogger.info("Done.")

    # `IFSystem` object that contains a list of components
    rootLogger.info("Building infrastructure system model... ")
    infrastructure, algorithm_factory = ingest_model(config)

    # assign the algorithm factory to the scenario
    scenario.algorithm_factory = algorithm_factory

    sys_topology_view = SystemTopology(infrastructure, scenario)
    sys_topology_view.draw_sys_topology(viewcontext="as-built")
    rootLogger.info("Done.")

    rootLogger.info("Initiating model run...")

    post_processing_list = calculate_response(scenario, infrastructure)
    # After the response has been calculated the post processing
    # will record the results
    post_processing(infrastructure, scenario, post_processing_list)
    rootLogger.info("Done.")
