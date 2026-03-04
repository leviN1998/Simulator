""" This file gives an example usage of how to run the simulation pipeline.

    If parralelization or batch processing is desired, i recommend to have a look at the main simulation loops of both classes and redirect the outputs to files.
    It would be important as well to handel the problem of a folder with images inside. (data/tmp). Maybe use one folder per thread or anything like that.
    The yaml config can be used like a dict and parameters can be changed there. With this, a different trajectory is easy to implemetn by just changing a few values.
"""
import yaml
import sys
sys.path.append("../src")
sys.path.append("../src/IEBCS")
sys.path.append("../src/utils")
from blender_simulator import BlenderSimulator
from event_simulator import EventSimulator

#load config
with open("../config/default.yaml", "r") as f:
    config = yaml.safe_load(f)

# run blender simulation
simulator = BlenderSimulator(config)
simulator.run_simulation()

# run event simulation
simulator = EventSimulator(config)
simulator.run_simulation()