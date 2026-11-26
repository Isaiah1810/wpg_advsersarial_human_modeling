import pandas as pd
from inferInvars import inferInvariants
from transProbs import transitionProbabilities

def toSymbolic(explt, states):
	invars = inferInvariants(explt)
	trans = transitionProbabilities(explt, invars, states)
	return trans
