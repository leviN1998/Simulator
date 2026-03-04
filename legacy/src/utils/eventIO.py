''' File to contain custom event saving and loading functions

    This file contains functions to handle event-Buffers from IEBCS and
    save/load them as hdf5 files.

    It also contains tools to get information about tghe event content of a buffer,
    such as the total time passed, or the number of events and other useful stuff.

    A function to create a video from the events is also included.
'''

import sys
import h5py
import hdf5plugin
import cv2
import numpy as np
import tqdm
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# import the IECBS simulator
# from src.utils.IEBCS.event_buffer import EventBuffer
sys.path.append("src/utils/IEBCS")
sys.path.append("src/utils")
from event_buffer import EventBuffer
import event_representations

mv_dtype = {'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}


'''

events.h5
├── events
│   ├── polarity       → int8 or bool, shape (N,)
│   ├── x              → uint16 or int, shape (N,)
│   ├── y              → uint16 or int, shape (N,)
│   └── timestamp      → uint64 or float64, shape (N,)
└── metadata (attrs)
    ├── resolution     → (width, height)
    └── source         → "DAVIS346", "CustomCam", etc.

'''

mv_dtype = np.dtype({'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16})


def buffer_to_array(event_buffer: EventBuffer) -> np.ndarray:
    """ Convert the event buffer to a structured numpy array
        Args:
            event_buffer: EventBuffer to convert
        Returns:
            Structured numpy array containing the events
    """
    return np.array(list(zip(event_buffer.get_x(), event_buffer.get_y(),
                               event_buffer.get_p(), event_buffer.get_ts())),
                    dtype=mv_dtype)


def buffer_to_video(event_buffer: EventBuffer, tw_us: int, sensor_size: tuple[int, int] = (1280, 720)) -> list[np.ndarray]:
    """ Convert the event buffer to a list of frames (numpy arrays)
        Args:
            event_buffer: EventBuffer to convert
        Returns:
            List of frames (numpy arrays)
    """
    frames = []
    it = EventIterator(buffer_to_array(event_buffer), tw_us=tw_us)

    empty = 0
    frames_cnt = 0
    for evs in it:
        if len(evs) == 0:
            img = np.zeros((event_buffer.get_y().max()+1, event_buffer.get_x().max()+1, 3), dtype=np.uint8)
            empty += 1
        
        else:
            ev_frame = event_representations.events_to_image(evs["x"], evs["y"], evs["t"], evs["p"], sensor_size=sensor_size)
            img = event_representations.ev_frame_to_img(ev_frame)
        
        frames.append(img)
        frames_cnt += 1
        # if frames_cnt % 100 == 0:
        #     print(f"Converted {frames_cnt} frames, {empty} were empty")
    return frames


def save_hdf5_old(event_buffer: EventBuffer, filename: str):
    """ Save the event buffer as hdf5 file
        Args:
            event_buffer: EventBuffer to save
            filename: name of the file to save to
    """
    with h5py.File(filename, 'w') as f:
        grp = f.create_group('events')
        grp.create_dataset('p', data=event_buffer.get_p())
        grp.create_dataset('x', data=event_buffer.get_x())
        grp.create_dataset('y', data=event_buffer.get_y())
        grp.create_dataset('t', data=event_buffer.get_ts())

        f.attrs['resolution'] = (5, 5) # TODO: add good attributes


def save_hdf5(event_buffer: EventBuffer, filename: str, bias, resolution=(1280, 720), chunk_size=10_000_000, 
                compression=hdf5plugin.Blosc(cname="zstd", clevel=1, shuffle=hdf5plugin.Blosc.BITSHUFFLE),
                clevel=1, sensor="IECBS Simulator"):
    """ Save the event buffer as hdf5 file as specified by David
        Args:
            event_buffer: EventBuffer to save
            filename: name of the file to save to
            bias: Bias values (simulator) to save in the file. (th_pos, th_neg, th_n, lat, tau, jit, bgn, refp)
            chunk_size (int): Number of events per chunk.
            compression: Compression algorithm settings.
            clevel (int): Compression level.
    """
    # sensor = "IECBS Simulator"
    width, height = resolution
    with h5py.File(filename, 'w') as f:
        event_group = f.create_group('events')
        dset_x = event_group.create_dataset("x", shape=(0,), maxshape=(None,), dtype="uint16",
                                            chunks=(chunk_size,), compression=compression, compression_opts=clevel)
        dset_y = event_group.create_dataset("y", shape=(0,), maxshape=(None,), dtype="uint16",
                                            chunks=(chunk_size,), compression=compression, compression_opts=clevel)
        dset_p = event_group.create_dataset("p", shape=(0,), maxshape=(None,), dtype="uint8",
                                            chunks=(chunk_size,), compression=compression, compression_opts=clevel)
        dset_t = event_group.create_dataset("t", shape=(0,), maxshape=(None,), dtype="uint64",
                                            chunks=(chunk_size,), compression=compression, compression_opts=clevel)
        
        f.create_dataset("t_offset", data=[0], maxshape=(None,))
        f.attrs.update({"width": width, "height": height, "sensor": sensor})
        f.create_dataset("bias", data=bias, maxshape=(None,)) # TODO: add good attributes

        # save the events
        dset_x.resize((event_buffer.i,))
        dset_y.resize((event_buffer.i,))
        dset_p.resize((event_buffer.i,))
        dset_t.resize((event_buffer.i,))
        dset_x[:] = event_buffer.get_x()
        dset_y[:] = event_buffer.get_y()
        dset_p[:] = event_buffer.get_p()
        dset_t[:] = event_buffer.get_ts()

        ms_to_idx = generate_ms_to_idx(dset_t[:])
        dset_ms = f.create_dataset("ms_to_idx", shape=(len(ms_to_idx),), maxshape=(None,), dtype="uint64")
        dset_ms[:] = ms_to_idx



def generate_ms_to_idx(timestamps, last_index=0, previous_time_stamps=0):
    """
    Generate an optimized mapping of milliseconds to event indices.

    Args:
        timestamps (np.ndarray): Array of event timestamps.
        last_index (int): Starting index for ms_to_idx array.
        previous_time_stamps (int): Offset for previous timestamps.

    Returns:
        np.ndarray: Array mapping milliseconds to event indices.
    """
    if timestamps.size == 0:
        return np.array([], dtype=np.int64)

    timestamps_ms = timestamps // 1_000
    unique_ms, first_indices = np.unique(timestamps_ms, return_index=True)

    max_time = unique_ms[-1] if unique_ms.size > 0 else last_index
    ms_to_idx = np.zeros(int(max_time + 1 - last_index), dtype=np.int64)

    ms_to_idx[unique_ms - last_index] = first_indices + previous_time_stamps
    return replace_zeros(ms_to_idx)


def replace_zeros(arr):
    """
    Replaces zero values within the timestamps.

    Args:
        arr (np.ndarray): Array mapping milliseconds to event indices.

    Returns:
        np.ndarray: Cleaned array with filled zero entries.
    """
    mask = arr == 0
    if mask.sum() == 0:
        return arr
    if arr.sum() == 0:
        return arr

    mask[0] = 0
    valid_idx = np.where(~mask)[0]
    valid_values = arr[valid_idx]
    next_nonzero_idx = np.searchsorted(valid_idx, np.where(mask)[0])
    arr[mask] = valid_values[next_nonzero_idx]

    return arr



def load_hdf5(filename: str) -> EventBuffer:
    """ Load the event buffer from an hdf5 file
        Args:
            filename: name of the file to load from
        Returns:
            EventBuffer with the loaded events
    """
    buf = EventBuffer(0)
    with h5py.File(filename, 'r') as f:
        data = f['events']
        buf.x = data['x'][:]
        buf.y = data['y'][:]
        buf.p = data['p'][:]
        buf.ts = data['t'][:]
        buf.i = buf.ts.shape[0]
        buf.ms_to_idx = f.get("ms_to_idx", np.array([], dtype=np.uint64))[:]

    return buf


def load_hdf5_metavision(filename: str) -> EventBuffer:
    """ Load the event buffer from an hdf5 file created by metavision studio
        The file structure is different, therefore we need this function
        file
           |- CD
           |   |- events
           |       |- x
           |       |- y
           |       |- p
           |       |- t
           |- Indexes

        Args:
            filename: name of the file to load from
        Returns:
            EventBuffer with the loaded events
    """
    buf = EventBuffer(0)
    with h5py.File(filename, 'r') as f:
        events = f['CD']["events"]
        buf.x = events['x'][:]
        buf.y = events['y'][:]
        buf.p = events['p'][:]
        buf.ts = events['t'][:]
        buf.i = buf.ts.shape[0]

    return buf


def save_hdf5_metavision(event_buffer: EventBuffer, filename: str, bias, resolution=(1280, 720), chunk_size=10_000_000, 
                compression=hdf5plugin.Blosc(cname="zstd", clevel=1, shuffle=hdf5plugin.Blosc.BITSHUFFLE),
                clevel=1):
    """ Save the event buffer as hdf5 file as specified by Metavision
        Args:
            event_buffer: EventBuffer to save
            filename: name of the file to save to
            bias: Bias values (simulator) to save in the file. (th_pos, th_neg, th_n, lat, tau, jit, bgn, refp)
            chunk_size (int): Number of events per chunk.
            compression: Compression algorithm settings.
            clevel (int): Compression level.
    """
    sensor = "IECBS Simulator"
    width, height = resolution
    with h5py.File(filename, 'w') as f:
        cd_group = f.create_group('CD')
        event_group = cd_group.create_dataset('events', shape=(0,), maxshape=(None,), dtype=mv_dtype,
                                              chunks=(chunk_size,), compression=compression, compression_opts=clevel)
        indexes = cd_group.create_dataset("indexes", shape=(0,), maxshape=(None,), dtype=[('id', '<u8'), ('ts', '<i8')],
                                           chunks=(chunk_size,), compression=compression, compression_opts=clevel)
        ext_group = f.create_group('EXT_TRIGGER')
        _ = ext_group.create_dataset('events', shape=(0,), maxshape=(None,), dtype={'names': ['p', 't', 'id'], 'formats': ['<i2', '<i8', '<i2'], 'offsets': [0, 8, 16], 'itemsize': 24},
                                     chunks=(chunk_size,), compression=compression, compression_opts=clevel)
        _ = ext_group.create_dataset('indexes', shape=(0,), maxshape=(None,), dtype=[('id', '<u8'), ('ts', '<i8')],
                                     chunks=(chunk_size,), compression=compression, compression_opts=clevel)

        # dset_x = event_group.create_dataset("x", shape=(0,), maxshape=(None,), dtype="uint16",
        #                                     chunks=(chunk_size,), compression=compression, compression_opts=clevel)
        # dset_y = event_group.create_dataset("y", shape=(0,), maxshape=(None,), dtype="uint16",
        #                                     chunks=(chunk_size,), compression=compression, compression_opts=clevel)
        # dset_p = event_group.create_dataset("p", shape=(0,), maxshape=(None,), dtype="uint8",
        #                                     chunks=(chunk_size,), compression=compression, compression_opts=clevel)
        # dset_t = event_group.create_dataset("t", shape=(0,), maxshape=(None,), dtype="uint64",
        #                                     chunks=(chunk_size,), compression=compression, compression_opts=clevel)
        
        #f.create_dataset("t_offset", data=[0], maxshape=(None,))
        f.attrs.update({"width": width, "height": height, "sensor": sensor})
        # f.create_dataset("bias", data=bias, maxshape=(None,)) # TODO: add good attributes

        # save the events
        # dset_x.resize((event_buffer.i,))
        # dset_y.resize((event_buffer.i,))
        # dset_p.resize((event_buffer.i,))
        # dset_t.resize((event_buffer.i,))
        # dset_x[:] = event_buffer.get_x()
        # dset_y[:] = event_buffer.get_y()
        # dset_p[:] = event_buffer.get_p()
        # dset_t[:] = event_buffer.get_ts()
        
        event_group.resize((event_buffer.i,))
        event_group["x"] = event_buffer.get_x()
        event_group["y"] = event_buffer.get_y()
        event_group["p"] = event_buffer.get_p()
        event_group["t"] = event_buffer.get_ts()

        ms_to_idx = generate_ms_to_idx(event_group["t"])
        idxs = []
        for i in range(len(ms_to_idx)):
            if i % 2 == 0:
                idxs.append((ms_to_idx[i], i*1000))
        
        indexes.resize((len(idxs),))
        indexes["id"] = np.array(idxs)[:, 0]
        indexes["ts"] = np.array(idxs)[:, 1]

        


def print_event_info(event_buffer: EventBuffer):
    """ Print information about the event buffer
        Args:
            event_buffer: EventBuffer to print information about
    """
    print("Event Buffer Information:")
    print(f"Number of events: {event_buffer.i}")
    print(f"Time range [us]: {event_buffer.get_ts()[0]} - {event_buffer.get_ts()[-1]}")
    print(f"Total time [s]: {(event_buffer.get_ts()[-1] - event_buffer.get_ts()[0]) / 1e6:.6f}")
    total_rounds = 2.0
    print(f"Recalculated RPS (using {total_rounds} total rounds): {total_rounds / ((event_buffer.get_ts()[-1] - event_buffer.get_ts()[0]) / 1e6):.2f} RPS")
    print(f"Resolution: {np.max(event_buffer.x) + 1} x {np.max(event_buffer.y) + 1}")
    print(f"Polarity values: {np.unique(event_buffer.get_p())}")
    print(f"Events per pixel: {event_buffer.i / ((np.max(event_buffer.get_x()) + 1) * (np.max(event_buffer.get_y()) + 1)):.2f}")
    print(f"Sample events (first 10):")
    for i in range(min(10, event_buffer.i)):
        print(f"Event {i}: t={event_buffer.get_ts()[i]}, x={event_buffer.get_x()[i]}, y={event_buffer.get_y()[i]}, p={event_buffer.get_p()[i]}")



def create_video(events: EventBuffer, save_filename: str, resolution=(1280, 720), fps=30.0, tw=1000):
    """ Create a video from the events in an EventBuffer
        Args:
            events: EventBuffer with the events
            save_filename: name of the file to save the video to
            resolution: resolution of the video (width, height)
            fps: frames per second for the video
            tw: time window for each frame in ms
    """
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    
    ts = events.get_ts()
    x = events.get_x()
    y = events.get_y()
    p = events.get_p()

    res = [resolution[1], resolution[0]]
    out = cv2.VideoWriter(save_filename, fourcc, fps, (res[1], res[0]))
    img = np.zeros((res[0], res[1]), dtype=np.uint8)
    tsurface = np.zeros((res[0], res[1]), dtype=np.uint64)
    indsurface = np.zeros((res[0], res[1]), dtype=np.uint64)
    
    # for t in range(ts[0], ts[-1], tw):
    for t in tqdm.tqdm(range(ts[0], ts[-1], tw), desc="Creating video frames", unit="frame"):
        ind = np.where((ts > t)&(ts < t + tw))
        tsurface[:, :] = 0
        tsurface[y[ind], x[ind]] = t + tw
        indsurface[y[ind], x[ind]] = p[ind]
        ind = np.where(tsurface > 0)
        img[:, :] = 125
        img[ind] = 125 + (2 * indsurface[ind] - 1) * np.exp(-(t + tw - tsurface[ind].astype(np.float32))/ (tw/30)) * 125
        img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_c = cv2.applyColorMap(img_c, cv2.COLORMAP_VIRIDIS)
        
        out.write(img_c)
    out.release()


class EventIterator:
    """
    Iterates over an event-array by using custom time-windows
    
    """
    def __init__(self, events: np.ndarray, tw_us: int = 1000):
        self.events = events
        self.tw_us = tw_us
        self.t_min = events['t'].min()
        self.t_max = events['t'].max()
        self.current_time = self.t_min

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_time > self.t_max:
            raise StopIteration
        # Get the next chunk of events
        chunk_idxs = np.where((self.events['t'] >= self.current_time) & (self.events['t'] < self.current_time + self.tw_us))[0]
        chunk = np.empty(0, dtype=mv_dtype)

        if chunk_idxs.size > 0:
            chunk = self.events[chunk_idxs]
            
        self.current_time += self.tw_us
        return chunk
    

if __name__ == "__main__":
    path = "/home/lkolmar/Documents/metavision/recordings/calibration_mv.hdf5"
    
    output_path = "/home/lkolmar/Documents/metavision/recordings/calibration.hdf5"

    buf = load_hdf5_metavision(path)
    print_event_info(buf)
    save_hdf5(buf, output_path, bias=[0, 15, 15, 0 ,0, 0])