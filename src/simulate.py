""" Simulates n datapoints of the specified dataset

    This script needs a path with structure as specified in "create_sim_table.py"
    It loads the configurations and simulates n simulations.
    The class simulator.py is used for every individual run

"""

import numpy as np
import yaml
import pandas as pd
import sys
sys.path.append("src/utils/")
sys.path.append("src/simulator/")
sys.path.append("src/utils/IEBCS/")
import simulator
import rotations
import logger
import time
import os


path = "/data/lkolmar/datasets/realistic/"


if __name__ == "__main__":
    n = int(sys.argv[1])              # number of simulations to run
    offset = int(sys.argv[2])         # offset to start from
    pid = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # process id for logging

    with open(path + "config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    simulation_df = pd.read_csv(path + "config/simulation.csv")
    print(simulation_df[:100])

    print("Applying offset")
    print(simulation_df[offset:offset + 100])

    if n == 0:
        n = len(simulation_df) - offset
        print(f"Setting n to {n} (length of simulation_df - offset)")

    try:
        os.mkdir(path + "tmp/pid_" + str(pid) + "/")
    except FileExistsError:
        print(f"Temporary directory for pid {pid} already exists, using it.")
    basic_logger = logger.Logger(path=path + "tmp/pid_" + str(pid) + "/")
    basic_logger.info(f"Running {n} simulations with offset {offset}")

    for i in range(n):
        start_ts = time.time()
        index = i + offset
        basic_logger.info(f"Running simulation {i + 1} of {n} with index {index}")
        data = simulation_df.iloc[index]

        # load data
        initital_orientation = [data["initial_rot_x"], data["initial_rot_y"], data["initial_rot_z"]]
        spin = [data["rotation_x"], data["rotation_y"], data["rotation_z"]]
        ball_start = [data["ball_start_x"], data["ball_start_y"], data["ball_start_z"]]
        ball_end = [data["ball_end_x"], data["ball_end_y"], data["ball_end_z"]]
        scale_start = data["scale_start"]
        scale_end = data["scale_end"]
        simulation_time = data["simulation_time"]


        # put into config
        config["spin_axis"] = spin
        config["initial_orientation"] = initital_orientation
        config["ball_start"] = ball_start
        config["ball_end"] = ball_end
        config["scale_start"] = scale_start
        config["scale_end"] = scale_end
        config["simulation_time"] = simulation_time

        s = rotations.Rotation()
        s.set_axis(spin[0], spin[1], spin[2])
        rps = s.get_angle()
        total_rotations = rps * (simulation_time / 1e6)
        print(f"Total rotations for simulation {index}: {total_rotations}")
        print(f"Spin vector: {spin}, rps: {rps}, simulation_time: {simulation_time}")
        config["total_rotations"] = total_rotations

        if data["finished"] == True:
            basic_logger.info(f"Simulation {index} already finished, skipping.")
            continue

        # create simulator instance
        sim = simulator.Simulator(config, logger=basic_logger, simulation_nr=index, pid=pid)
        sim.run_simulation()
        simulation_df.loc[index, "finished"] = True
        simulation_df.loc[index, "path"] = f"data/{str(index).zfill(5)}/{str(index).zfill(5)}_"
        simulation_df.to_csv(path + f"config/simulation_pid{pid}.csv", index=False)
        duration_s = time.time() - start_ts
        estimated_time = (n - i - 1) * duration_s
        estimated_time /= 60
        estimated_time /= 60
        basic_logger.info(
            f"Simulation {index} finished in {time.time() - start_ts:.2f} seconds. "
            f"Estimated time left {int(estimated_time)}h {((estimated_time - int(estimated_time)) * 60):.1f}m."
        )

    basic_logger.info("All simulations finished.")
    basic_logger.close()
    print("Finished!")