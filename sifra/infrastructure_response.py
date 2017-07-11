import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import parmap

from model_ingest import ingest_spreadsheet
from sifraclasses import _ScenarioDataGetter
from sifra.modelling.hazard_levels import HazardLevels


def run_scenario(config_file):
    sc = _ScenarioDataGetter(config_file)

    infrastructure = ingest_spreadsheet(config_file)
    hazard_levels = HazardLevels(sc)
    response_array = []

    # Use the parallel option in the scenario to determine how
    # to run
    response_array = []
    response_array.extend(parmap.map(infrastructure.expose_to,
                                     hazard_levels.hazard_range(),
                                     parallel=False))

    print("response array len={}".format(len(response_array)))


def main():

    SETUPFILE = sys.argv[1]

    run_scenario(SETUPFILE)


if __name__ == '__main__':
    main()
