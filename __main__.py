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
    main()