import explicit
import argparse
import stormpy

def parse_arguments():
	parser = argparse.ArgumentParser(
		prog="MDP",
		description="Creates an MDP representing the Poacher from available data"
	)
	parser.add_argument('--input-dir', default='../outputs/', type=str, dest='directory',
		help="Input directory where simulation data is stored. Should be output location from running the game"
	)

	args = parser.parse_args()
	return args

def create_explicit_mdp(args):
	model = explicit.convert_to_explicit(args.directory)

def generalize_mdp():
	pass

def main():
	args = parse_arguments()
	create_explicit_mdp(args)

if __name__ == '__main__':
	main()
