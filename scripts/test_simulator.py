import sys
sys.path.append("src/")
from simulator import Simulator
from logger import Logger
import yaml
import time

# Full setup: 175s
# With border: 128s


if __name__ == "__main__":
    print("test simulator module")

    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    logger = Logger(path="/data/lkolmar/datasets/test/tmp/")

    sim = Simulator(config, logger, simulation_nr=1, pid=0)
    start_time = time.time()
    sim.run_simulation()
    end_time = time.time()
    print(f"Simulation finished. Time taken: {end_time - start_time} seconds")