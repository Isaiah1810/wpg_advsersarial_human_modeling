import pandas as pd
import py_trees

from wpg.game_env import game_env
from wpg.game_map import CellType
from wpg.gentypes import Coordinates
from wpg.rng import get_rng_manager

class Animal:
    def __init__(self, config, x, y):
        self.location = Coordinates(x, y)
        self.poached = False
        self.will_run = config['animal_run']
