""" File for manipulating raw event recordings and transforming them into different representations.

    Inputs are not EventBuffers, because the DataLoader shoul dbe able to work on Hdf5 for dynamic loading

    Inspired by https://github.com/TimoStoff/event_utils
"""

import numpy as np
import cv2


def events_to_image(xs, ys, ts, ps, sensor_size: tuple[int, int] = (1280, 720)) -> np.ndarray:
    """Convert an event buffer to an image representation.

    Args:
        xs (np.ndarray): The x-coordinates of the events.
        ys (np.ndarray): The y-coordinates of the events.
        ts (np.ndarray): The timestamps of the events.
        ps (np.ndarray): The polarities of the events. (could also be counts) -> this part will be visualized
        sensor_size (tuple[int, int]): The size of the sensor (width, height). Defaults to (1280, 720).

    Returns:
        np.ndarray: [x, y] The image representation of the events, where each pixel value corresponds to the normalized
    """
    assert len(xs) == len(ys) == len(ts) == len(ps), "All event arrays must have the same length."
    # Create an empty image
    image = np.zeros((sensor_size[1], sensor_size[0]), dtype=np.float32)
    max_p = ps.max()
    min_p = ps.min()
    # Normalize polarities to the range [0, 255]
    ps_normalized = ((ps - min_p) / (max_p - min_p) * 255).astype(np.uint8)
    # Map events to the image
    for x, y, p in zip(xs, ys, ps_normalized):
        # image[y, x] += p
        image[y, x] += 1
    return image


def get_voxel_grid_as_image(voxelgrid: np.ndarray) -> np.ndarray:
    """ Convert a voxel grid to an image representation.
    
        Args:
            voxelgrid (np.ndarray): [B, x, y] The voxel grid to convert.
    
        Returns:
            np.ndarray: The image representation of the voxel grid.
    """
    images = []
    splitter = np.ones((voxelgrid.shape[1], 2))*np.max(voxelgrid)
    for time_bin in voxelgrid:
        # images.append(time_bin / np.max(time_bin))
        images.append(time_bin)
        images.append(splitter)
    
    images.pop()
    sidebyside = np.hstack(images)
    # sidebyside = cv2.normalize(sidebyside, None, 0, 255, cv2.NORM_MINMAX)
    return sidebyside



def events_to_voxel(xs, ys, ts, ps, num_bins: int = 10, sensor_size: tuple[int, int] = (1280, 720)) -> np.ndarray:
    """ Convert an event buffer to a voxel representation.
    
        Important: This function converts all events ointo a single voxel grid.
        For a whole simulation / video this should be used on every frame
        This will be the input to the model at every recurrent time step.

        -> It is a good question how the individual voxels should look like
           In this case we just add all events and dont take their polarity into account.
    
    Args:
        xs (np.ndarray): The x-coordinates of the events. (All events need to be sorted by time)
        ys (np.ndarray): The y-coordinates of the events.
        ts (np.ndarray): The timestamps of the events.
        ps (np.ndarray): The polarities of the events.
        num_bins (int): The number of bins for the voxel grid. Defaults to 10.
        sensor_size (tuple[int, int]): The size of the sensor (width, height). Defaults to (1280, 720).
    Returns:
        np.ndarray: [B, x, y] The voxel grid representation of the events.
    """
    assert len(xs) == len(ys) == len(ts) == len(ps), "All event arrays must have the same length."

    # Create the voxel grid
    voxel_grid = np.zeros((num_bins, sensor_size[1], sensor_size[0]),dtype=np.float32)
    t0 = ts.min()
    t1 = ts.max()
    bin_size = (t1 - t0) / num_bins
    img_size = (sensor_size[0]+1, sensor_size[1]+1)
    
    for b in range(num_bins):
        # Get the events in the current bin
        mask = (ts >= t0 + b * bin_size) & (ts < t0 + (b + 1) * bin_size)
        xs_bin = xs[mask]
        ys_bin = ys[mask]
        ps_bin = ps[mask]

        coords = np.stack((ys_bin, xs_bin))
        try:
            abs_coords = np.ravel_multi_index(coords, img_size)
        except ValueError:
            print("Issue with input arrays! minx={}, maxx={}, miny={}, maxy={}, coords.shape={}, \
                    sum(coords)={}, sensor_size={}".format(np.min(xs), np.max(xs), np.min(ys), np.max(ys),
                    coords.shape, np.sum(coords), img_size))
            raise ValueError
        v = np.bincount(abs_coords, minlength=img_size[0] * img_size[1])
        v = v.reshape(img_size)
        voxel_grid[b] = v[0:sensor_size[0], 0:sensor_size[1]]
        voxel_grid[b] = (voxel_grid[b] / voxel_grid[b].max() * 255).astype(np.float32) if voxel_grid[b].max() > 0 else voxel_grid[b]

    return voxel_grid


# TODO: think about neg/pos voxels
# TODO: check if dataloading is fast enought -> TimoStoff has some torch dataloading stuff


def create_sequence(events, time_window=500, num_bins=10, sensor_size=(100, 100), flip=False) -> np.ndarray:
    """ Create a sequence of voxel grids from events.

    This is usefull for preprocessing data such that e.g. FireNet can use it as input
    
    Args:
        events (np.ndarray): The event data structured as a numpy array with fields ['x', 'y', 't', 'p'].
        time_window (int): The time window in microseconds for each frame.
        num_bins (int): The number of bins for the voxel grid.
        sensor_size (tuple[int, int]): The size of the sensor (width, height).
    
    Returns:
        np.ndarray: A sequence of voxel grids, where each voxel grid corresponds to a time window.
        The shape of the output is [num_frames, num_bins, sensor_size[0], sensor_size[1]].
    """
    start_time = events['t'][0]
    end_time = events['t'][-1]

    # Vectorized version using numpy
    t = events['t']
    # Compute the bin index for each event
    bin_indices = ((t - start_time) // time_window).astype(int)
    num_frames = ((end_time - start_time) // time_window) + 1
    sequences = np.zeros((num_frames, num_bins, sensor_size[0], sensor_size[1]), dtype=np.float32)

    for i in range(num_frames):
        mask = bin_indices == i
        seq = events[mask]
        if len(seq) > 0:
            sequences[i] = events_to_voxel(seq['x'], seq['y'], seq['t'], seq['p'], num_bins=num_bins, sensor_size=sensor_size)
            if flip:
                sequences[i] = np.flip(sequences[i], axis=(1,2))

    return sequences



def ev_frame_to_img(f):
    """ Function to create pretty images for video representation
    """
    norm_frame = ((f / 12) * 255).clip(0, 255).astype(np.uint8)
    img = np.zeros((norm_frame.shape[0], norm_frame.shape[1], 3), dtype=np.uint8)
    # Set background to dark blue
    img[:, :, 2] = 40  # B
    img[:, :, 1] = 0   # G
    img[:, :, 0] = 0   # R
    # Set event pixels to light blue
    mask = norm_frame > 0
    # print(norm_frame.shape, frame.shape, mask.shape, img.shape)
    img[mask, 2] = 220  # B
    img[mask, 1] = 180  # G
    img[mask, 0] = 100  # R
    return img