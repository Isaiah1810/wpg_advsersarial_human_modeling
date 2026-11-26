import explicit
import generalize
import argparse
from pathlib import Path
import stormpy

def parse_arguments():
	parser = argparse.ArgumentParser(
		prog="MDP",
		description="Creates an MDP representing the Poacher from available data"
	)
	parser.add_argument('--input-dir', default='../outputs/', type=str, dest='directory',
		help="Input directory where simulation data is stored. Should be output location from running the game"
	)
	parser.add_argument('--output', default='poacher.nm', type=str, dest='output',
		help="Path of the PRISM model file to write (.nm = PRISM MDP)"
	)
	parser.add_argument('--module-name', default='poacher', type=str, dest='module_name',
		help="Name of the PRISM module"
	)
	parser.add_argument('--no-check', action='store_true', dest='no_check',
		help="Skip parsing/building the written model with stormpy"
	)

	args = parser.parse_args()
	return args

def create_explicit_mdp(args):
	model = explicit.convert_to_explicit(args.directory)
	return model

def get_state_space():
	stateSpace = explicit.get_state_space()
	return stateSpace

def generalize_mdp(explicit):
	states = get_state_space()
	symb = generalize.toSymbolic(explicit,states)
	return symb

###############################
# write prism
# arguments:
#	symb		TransitionModel from transitionProbabilities
#	path		output file (.nm)
#	moduleName	name of the PRISM module
# output:
#	path		the file written
###############################
def write_prism(symb, path, moduleName):
	text = symb.to_prism(module_name=moduleName)
	path = Path(path)
	path.write_text(text, encoding="utf-8")
	return path

###############################
# check prism
# parses the written file with storm and builds the MDP, so syntax errors,
# out-of-range updates and bad probabilities are caught right away
###############################
def check_prism(path):
	program = stormpy.parse_prism_program(str(path))
	model = stormpy.build_model(program)
	print(f"storm: {model.nr_states} states, {model.nr_transitions} transitions, "
		  f"{model.nr_choices} choices")
	return model

def main():
	args = parse_arguments()
	emdp = create_explicit_mdp(args)
	smdp = generalize_mdp(emdp)
	print(smdp.table().to_string())          # per-action outcome probabilities
	out = write_prism(smdp, args.output, args.module_name)
	print(f"wrote {out.resolve()}")
	if not args.no_check:
		check_prism(out)

if __name__ == '__main__':
	main()
