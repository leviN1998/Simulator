import os
import sys
import numpy as np
import pandas as pd
import shutil

sys.path.append("src/utils/")
import rotations

example_path = "data/datasets/example_dataset/"

def make_folder_structure(path:str) -> None:
    """ Creates the folder structure for the dataset

        Args:
            path (str): Path to the dataset folder
    """
    try:
        os.mkdir(path + "config")
        os.mkdir(path + "data")
        os.mkdir(path + "noise")
        os.mkdir(path + "tmp")
    except FileExistsError:
        print("One of the folders exists already, or couldn't be created. Please fix!")
        sys.exit()

    # create dummy files
    try:
        shutil.copy(example_path + "config/config.yaml", path + "config/config.yaml")
        shutil.copy(example_path + "config/readme.md", path + "config/readme.md")
        shutil.copy(example_path + "config/scene.blend", path + "config/scene_PLACEHOLDER.blend")
        shutil.copy(example_path + "noise/noise_neg_0.1lux.npy", path + "noise/noise_neg_0.1lux.npy")
        shutil.copy(example_path + "noise/noise_neg_3klux.npy", path + "noise/noise_neg_3klux.npy")
        shutil.copy(example_path + "noise/noise_neg_3klux.mat", path + "noise/noise_neg_3klux.mat")
        shutil.copy(example_path + "noise/noise_neg_161lux.npy", path + "noise/noise_neg_161lux.npy")
        shutil.copy(example_path + "noise/noise_pos_0.1lux.npy", path + "noise/noise_pos_0.1lux.npy")
        shutil.copy(example_path + "noise/noise_pos_3klux.npy", path + "noise/noise_pos_3klux.npy")
        shutil.copy(example_path + "noise/noise_pos_3klux.mat", path + "noise/noise_pos_3klux.mat")
        shutil.copy(example_path + "noise/noise_pos_161lux.npy", path + "noise/noise_pos_161lux.npy")

    except Exception as e:
        print(f"One of the files could not be created. Please fix! Error: {e}")


def angle_to_axis(vx, vy, vz, zhat):
    norm = np.sqrt(vx*vx + vy*vy + vz*vz)
    if norm == 0:
        return np.nan
    # Skalarprodukt durch Norm => cos(theta)
    cos_theta = (vx*zhat[0] + vy*zhat[1] + vz*zhat[2]) / norm
    # numerische Stabilisierung
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(abs(cos_theta)))

def is_top_or_backspin(axis: np.ndarray, threshold_angle: float, reference_axis: np.ndarray = np.array([1, 0, 0])) -> bool:
    """ Check if the rotation axis is top or backspin

        This function checks if the rotation axis is within the threshold angle to the reference axis.
        The reference axis is by default the x-axis, which means that a top or backspin is around the x-axis.

        Args:
            axis (np.ndarray): Rotation axis to check"""
    return angle_to_axis(axis[0], axis[1], axis[2], reference_axis) <= threshold_angle


def create_rotations(n:int, max_speed:float=80, min_speed:float=5) -> np.ndarray:
    """ Creates Rotations in a cubic way, as discussed with David

        This script creates a 3D cube of points that are interpreted as rotaitons (direction and magnitude).
        All rotations are scaled to be inside the range of max_speed and min_speed [rps]

        This script can also be used to check how many rotations would be generated with the current specs

        Args:
            n (int): points per axis. The cube will contain n³ points. The oucoming array will be smaller than that, depending on the other params
            max_speed (float): Maximum speed that should be generated. Unit should be rps
            min_speed (float): Minimum speed that are roation can have. Unit is rps

        Returns:
            rotations (np.ndarray): Numpy array containint all rotations
    """

    lin = np.linspace(-1, 1, n)
    x, y, z = np.meshgrid(lin, lin, lin)
    points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # cut out speeds that are not needed
    distances = np.linalg.norm(points, axis=1)
    # points = points[(distances <= 1) & (distances >= (min_speed / max_speed))]
    points = points[(distances <= 1) & (distances >= (0.9))]

    # select topspin and backspin
    threshold_angle = 20  # deg
    mask = np.array([is_top_or_backspin(p, threshold_angle) for p in points])
    points = points[mask]

    spins = []
    for p in points:
        p = p / np.linalg.norm(p)  # normalize
        for speed in range(int(min_speed), int(max_speed) + 1, 2):
            spins.append(p * speed)

    # points = points * max_speed

    return np.array(spins)



def create_initial_orientation_topspin(n:int, max_angle: float, min_angle: float) -> np.ndarray:
    """ Creates initial orientations for topspin

        This function creates random initial orientations that are within the given angle range.
        For the top / backspin dataset, the initial orientation axis is totally random, but the angle will be
        between -80 deg and +80 deg. With this configuration the logo is always visible in the video.

        Args:
            n (int): Number of initial orientations to generate
            max_angle (float): Maximum angle in degrees for the initial orientation
            min_angle (float): Minimum angle in degrees for the initial orientation

        Returns:
            np.ndarray: Array of shape (n, 3) containing the initial orientations (The angle is contained as vector length (degrees))
    """
    orientations = np.zeros((n, 3))
    for i in range(n):
        axis = rotations.random_rotation().get_axis()
        while np.linalg.norm(axis) <= 0.01:
            axis = rotations.random_rotation().get_axis()

        angle_deg = np.random.uniform(min_angle, max_angle)
        axis *= angle_deg / np.linalg.norm(axis)
        orientations[i] = axis

    return orientations

def create_table():
    """ Create the table for the newest dataset

    This dataset is tuned to be as close as possible to the real event data.
    
    """
    path = "/data/lkolmar/datasets/realistic_topspin/"
    try:
        os.mkdir(path)
    except FileExistsError:
        print("Folder exists already, or couldn't be created. Please fix!")
        sys.exit()

    make_folder_structure(path)

    samples = create_rotations(30, max_speed=140, min_speed=10)
    print(f"Created {len(samples)} rotations")

    initial_orientations = create_initial_orientation_topspin(len(samples), max_angle=80, min_angle=-80)
    print(f"Created {len(initial_orientations)} initial orientations")
    # start pos
    start_pos = np.column_stack([
        np.zeros(len(samples)),
        np.full(len(samples), -0.45),
        np.random.uniform(-0.25, 0.25, len(samples))
    ])
    # end pos
    end_pos = np.column_stack([
        np.zeros(len(samples)),
        np.full(len(samples), 0.45),
        np.random.uniform(-0.25, 0.25, len(samples))
    ])
    print(f"Created {start_pos.shape} start and end positions")
    print(f"Start pos example: {start_pos[0]}, End pos example: {end_pos[0]}")
    # scale_start 2.3
    scale = []
    for i in range(len(samples)):
        s = 2.3 + np.random.uniform(-0.1, 0.1)
        scale.append([s, s + np.random.uniform(-0.1, 0.1)])
    scale = np.array(scale)
    print(f"Created {scale.shape} scales")
    print(f"Scale example: {scale[0]} (start, end)")
    # simulation time 450000 us
    base_time = 450000  # us
    sim_times = np.random.uniform(low=base_time * 0.9, high=base_time * 1.1, size=len(samples))
    print(f"Created {sim_times.shape} simulation times")
    print(f"Simulation time example: {sim_times[0]} us")

    df = pd.DataFrame({
        'index': np.arange(len(samples)),
        'rotation_x': samples[:, 0],
        'rotation_y': samples[:, 1],
        'rotation_z': samples[:, 2],
        'initial_rot_x': initial_orientations[:, 0],
        'initial_rot_y': initial_orientations[:, 1],
        'initial_rot_z': initial_orientations[:, 2],
        'ball_start_x': start_pos[:, 0],
        'ball_start_y': start_pos[:, 1],
        'ball_start_z': start_pos[:, 2],
        'ball_end_x': end_pos[:, 0],
        'ball_end_y': end_pos[:, 1],
        'ball_end_z': end_pos[:, 2],
        'scale_start': scale[:, 0],
        'scale_end': scale[:, 1],
        'simulation_time': sim_times.astype(int),
        'finished': False,
        'path': "not set"
    })

    file_path = path + "config/simulation.csv"
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
   create_table()