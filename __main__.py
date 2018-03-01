# usage: python sifra -v -f "sample_structure.conf"

from sifra.infrastructure_response import run_scenario
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-infile","-f", type=str,
                        help="enter configuration file path.")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()

    if args.verbose:
        print("verbosity turned on")

    if args.infile:
        print(args.infile)


if __name__ == "__main__":
    # SETUPFILE = "C:\\Users\\u12089\\Desktop\\sifra-v0.2.0\\simulation_setup\\test_scenario_ps_coal.conf"

    SETUPFILE = "C:\\Users\\u12089\\Desktop\\sifra-v0.2.0\\tests\\test_simple_series_struct_dep.conf"

    run_scenario(SETUPFILE)