import stormpy
import numpy as np
from pathlib import Path
import pandas as pd
import json

def get_state_space():
	with open("config.json", "r", encoding="utf-8") as file:
		config = json.load(file)
	out: Dict[str, object] = {}
	for name, spec in config.items():
		if spec["type"] == "bool":
			out[name] = bool
		else:
			lo, hi = spec["range"]
			out[name] = (int(lo), int(hi))
	return out

###############################
# convert to explicit
# arguments:
#	fileDir		directory where raw output is stored
# output:
#	model		explicit MDP model of poacher
###############################
def convert_to_explicit(fileDir):
	# state space of model:
	# ( distance, seesDrone, hiding, usingJammer, action )
	simData = pd.DataFrame({"distance":[], "seesDrone":[], "hiding":[], "usingJammer":[], "action":[]})
	# outputs/ contains folders of dates
	for datePath in Path(fileDir).iterdir():
		# date folders contain paths based on time
		for timePath in Path(datePath).iterdir():
			# check that the path has correct information to read from
			if valid_path(timePath):
				# for each path, read in data files
				simData = pd.concat([simData,read_in_file(timePath)])
			# end if valid path
		# end for time path
	# end for date path
	for col in ['seesDrone', 'hiding', 'usingJammer']:
		simData[col] = simData[col].astype(bool)
	for col in ['distance']:
		simData[col] = simData[col].astype(int)
	#simData = simData.reset_index(drop=True)
	return simData
# end function read in files

###############################
# valid path
# arguments:
#	dirt		directory to validate
# outputs:
#	simDat		boolean; true iff data is valid
###############################
def valid_path(dirt):
	needed_data = ["drone", "poacher", "map"] # other files ("game") are not needed
	for needed in needed_data:
		file = dirt / f"{needed}_data.csv"
		if not file.exists():
			return False
		# end if file doesn't exist
	# end for each file
	return True
# end read in file

###############################
# read in file
# arguments:
#	dirt		directory to read simulation data
# outputs:
#	simDat		simulation data as pd dataframe
###############################
def read_in_file(dirt):
	# open csv information file in dataframes
	drone = pd.read_csv(dirt/'drone_data.csv')
	poach = pd.read_csv(dirt/'poacher_data.csv')
	# open map as 2D array
	with open(dirt/'map_data.csv') as fid:
		raw = fid.readlines()
	# end opening file
	map2D = [ list(x.strip().replace(',','')) for x in raw ]

	# convert to necessary headers
	# > compute distance
	drone["poach x"] = poach["x"]
	drone["poach y"] = poach["y"]
	drone["distance"] = np.ceil( np.sqrt( drone["poach x"]**2 + drone["poach y"]**2 ) ).astype(int)
	# > move drone detection in poach to seeing drone in drone
	drone["seesDrone"] = poach["drone_detected"]
	# > rename visible to hiding
	drone["hiding"] = drone.apply(lambda row: map2D[int(row["poach x"])][int(row["poach y"])] == '0', axis=1)
	# > rename to using Jammer
	drone["usingJammer"] = ~drone["gps_available"]
	# get action of poacher
	drone["action"] = drone["poacher_action"]

	# get subset of dataframe necessary
	colsToDrop = drone.columns.tolist()
	neededHeaders = ["usingJammer","hiding","distance","seesDrone","action"]
	for head in neededHeaders:
		colsToDrop.remove(head)
	simDat = drone.drop(columns=colsToDrop)
	# return simulation data
	return simDat
# end read in file
