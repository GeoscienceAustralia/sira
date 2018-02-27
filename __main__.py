# usage: python sifra -v -f "sample_structure.conf"

import logging
from colorama import Fore, Back, Style
from sifra.configuration import Configuration
from sifra.scenario import Scenario

formatter = '%(levelname)-8s %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter)

def main():


    # Construct the scenario object
    logging.info(Style.BRIGHT + Fore.GREEN + "Loading scenario config... " + Style.RESET_ALL)

    config_file_path = 'C:/Users/u12089/Desktop/sifra-dev/simulation_setup/config.json'
    config = Configuration(config_file_path)

    scenario = Scenario(config)


if __name__ == "__main__":

    main()
