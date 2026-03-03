"""
Script to run the simulation once with parameters specified in the configs/default.yaml file.

"""
import sys
import yaml
sys.path.append("../src/utils/")
sys.path.append("../src/")
sys.path.append("../src/IEBCS/")
import simulator

# To byass the logger, define a very basic one
class Logger:
    def __init__(self):
        pass

    def error(self, message: str):
        print(f"ERROR: {message}")

    def info(self, message: str):
        print(f"INFO: {message}")

    def debug(self, message: str):
        print(f"DEBUG: {message}")
        
    def progress(self, message: str):
        print(f"PROGRESS: {message}")

    def thread(self, message: str):
        print(f"THREAD: {message}")

    def close(self):
        pass

if __name__ == "__main__":
    number_of_simulations = 1 # just run once

    # load config
    with open("../configs/defaults.yaml", "r") as f:
        config = yaml.safe_load(f)

    basic_logger = Logger()
   
    sim = simulator.Simulator(config, logger=basic_logger, simulation_nr=0) # simulation_nr names the output folder, useful for batch simulations
    sim.run_simulation()

    basic_logger.close()