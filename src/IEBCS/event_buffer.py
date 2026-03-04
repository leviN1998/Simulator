import numpy
import numpy as np
from dat_files import write_event_dat
import hdf5plugin
import h5py


class EventBuffer():
    """ Structure to handle a buffer of dvs events """
    x = 0  # Array of x values
    y = 0  # Array of y values
    ts = 0  # Array of timestamps values (us)
    p = 0  # Array of polarity values (0 negative, 1 positive)
    i = 0  # Position of the next event

    def __init__(self, size):
        """ Resize the buffers
            Args:
                size: size of the new buffer, Minimum: 1
        """
        if size == 0:
            size = 1
        self.x = np.zeros(size, dtype=np.uint16)
        self.y = np.zeros(size, dtype=np.uint16)
        self.p = np.zeros(size, dtype=np.uint8)
        self.ts = np.zeros(size, dtype=np.uint64)
        self.i = 0

    def get_x(self):
        return self.x[:self.i]

    def get_y(self):
        return self.y[:self.i]

    def get_p(self):
        return self.p[:self.i]

    def get_ts(self):
        return self.ts[:self.i]

    def increase(self, nsize):
        """ Increase the size of a buffer to self.shape[0] size + nsize
            Args:
                nsize: number of free space elements to add
        """
        prev_shape = self.x.shape[0]
        x = np.zeros(prev_shape + nsize, dtype=np.uint16)
        y = np.zeros(prev_shape + nsize, dtype=np.uint16)
        p = np.zeros(prev_shape + nsize, dtype=np.uint8)
        ts = np.zeros(prev_shape + nsize, dtype=np.uint64)
        x[:prev_shape] = self.x
        y[:prev_shape] = self.y
        p[:prev_shape] = self.p
        ts[:prev_shape] = self.ts
        self.x = x
        self.y = y
        self.p = p
        self.ts = ts

    def remove_time(self, t_min, t_max):
        """
            Only keep events between t_min and t_max
        """
        ind = np.where((self.ts < t_min) | (self.ts > t_max))
        self.x = np.delete(self.x, ind)
        self.y = np.delete(self.y, ind)
        self.ts = np.delete(self.ts, ind)
        self.p = np.delete(self.p, ind)
        self.i = self.ts.shape[0]

    def remove_elt(self, nsize):
        """
            Remove the nsize first elements
        """
        if self.i - nsize < 0:
            nsize = self.i
        ind = np.arange(0, nsize, 1)
        self.x = np.delete(self.x, ind)
        self.y = np.delete(self.y, ind)
        self.ts = np.delete(self.ts, ind)
        self.p = np.delete(self.p, ind)
        self.i = self.i - nsize

    def remove_ev(self, p):
        """
            Remove the event at the position p
        """
        if self.i <= p:
            return
        self.x = np.delete(self.x, p)
        self.y = np.delete(self.y, p)
        self.ts = np.delete(self.ts, p)
        self.p = np.delete(self.p, p)
        self.i -= 1

    def remove_row(self, r, t):
        """
            Remove the event in row r at time t
        """
        if t == -1:
            ind = np.where((self.y == r) & (self.ts > 0))
        else:
            ind = np.where((self.y == r) & (self.ts < t) & (self.ts > 0))
        self.x = np.delete(self.x, ind)
        self.y = np.delete(self.y, ind)
        self.ts = np.delete(self.ts, ind)
        self.p = np.delete(self.p, ind)
        self.i -= ind[0].shape[0]

    def increase_ev(self, ev):
        """ Extend the event buffer with another event buffer
            If ev can be inserted into self, ev inserted, if not, increase the size of a buffer to original
            self.shape[0] + ev.shape[0]
            Args:
                ev: the EventBuffer added
            """
        if len(self.x) > 0 and not ev is None:
            if self.i + ev.x.shape[0] > self.x.shape[0] - 1:
                prev_shape = self.x.shape[0]
                x = np.zeros(prev_shape + ev.ts.shape[0], dtype=np.uint16)
                y = np.zeros(prev_shape + ev.ts.shape[0], dtype=np.uint16)
                p = np.zeros(prev_shape + ev.ts.shape[0], dtype=np.uint8)
                ts = np.zeros(prev_shape + ev.ts.shape[0], dtype=np.uint64)
                x[:self.i] = self.x[:self.i]
                y[:self.i] = self.y[:self.i]
                p[:self.i] = self.p[:self.i]
                ts[:self.i] = self.ts[:self.i]
                x[self.i:self.i + ev.x.shape[0]] = ev.x
                y[self.i:self.i + ev.x.shape[0]] = ev.y
                p[self.i:self.i + ev.x.shape[0]] = ev.p
                ts[self.i:self.i + ev.x.shape[0]] = ev.ts
                self.x = x
                self.y = y
                self.p = p
                self.ts = ts
            else:
                self.x[self.i:self.i + ev.i] = ev.x[:ev.i]
                self.y[self.i:self.i + ev.i] = ev.y[:ev.i]
                self.p[self.i:self.i + ev.i] = ev.p[:ev.i]
                self.ts[self.i:self.i + ev.i] = ev.ts[:ev.i]
            self.i += ev.i

    def copy(self, i1, ep, i2):
        """ Copy the i2 th event of the EventBuffer ep in to the i1 th position
            Args:
                i1: self will have a new event in i1
                ep: EventBuffer where the event comes from
                i2: i2th event from ep is takem
         """
        if i1 < len(self.x):
            self.x[i1] = ep.x[i2]
            self.y[i1] = ep.y[i2]
            self.ts[i1] = ep.ts[i2]
            self.p[i1] = ep.p[i2]
            self.i = i1 + 1

    def merge(self, ep1, ep2):
        """ Resize the EventBuffer and merge into the two EventBuffers ep1 nd ep2, sorted by their timestamps
            Args:
                ep1, ep2: eventBuffer
        """
        self.__init__(len(ep1.x) + len(ep2.x))
        i1 = 0
        i2 = 0
        for j in range(0, ep1.i + ep2.i, 1):
            if i1 == ep1.i:
                self.copy(j, ep2, i2)
                i2 += 1
            elif i2 == ep2.i:
                self.copy(j, ep1, i1)
                i1 += 1
            else:
                if ep1.ts[i1] < ep2.ts[i2]:
                    self.copy(j, ep1, i1)
                    i1 += 1
                else:
                    self.copy(j, ep2, i2)
                    i2 += 1
        self.i = ep1.i + ep2.i

    def sort(self):
        """ Sort the EventBuffer according to its timestamp """
        ind = np.argsort(self.ts[:self.i])
        self.ts[:self.i] = self.ts[:self.i][ind]
        self.x[:self.i] = self.x[:self.i][ind]
        self.y[:self.i] = self.y[:self.i][ind]
        self.p[:self.i] = self.p[:self.i][ind]

    def add(self, ts, y, x, p):
        """
            Add an event (ts, x, y, p) to the EventBuffer (push strategy)
            If y == -1, if means that x[0] contains the x position and x[1] the y position.
            Args:
                ts, y, x, p: new event array
        """
        if self.x.shape[0] == self.i:
            self.increase(1000)
            self.add(ts, y, x, p)
        else:
            self.ts[self.i] = ts
            self.x[self.i] = x
            self.y[self.i] = y
            self.p[self.i] = p
            self.i += 1

    def add_array(self, ts, y, x, p, inc=1000):
        """
            Add n events (ts, x, y, p) to the EventBuffer (push strategy)
            Args:
                ts, y, x, p: new event array
                inc: increment size
        """
        s = len(ts)
        if s > len(self.ts) - self.i:
            self.increase(inc)
            self.add_array(ts, y, x, p)
        else:
            self.ts[self.i:self.i + s] = ts
            self.x[self.i:self.i + s] = x
            self.y[self.i:self.i + s] = y
            self.p[self.i:self.i + s] = p
            self.i += s

    def write(self, filename, width=None, height=None):
        """ Write the events into a .dat file
            Args:
                filename: path of the file
        """
        # sort events to have a monotonically timestamps
        self.sort()
        write_event_dat(filename, self.ts[:self.i], self.x[:self.i], self.y[:self.i], self.p[:self.i],
                        event_type='dvs', width=width, height=height)


    def save_hdf5(self, filename: str, bias, width: int, height: int,
                  chunk_size: int = 10_000_000,
                  compression=hdf5plugin.Blosc(cname='zstd', clevel=1, shuffle=hdf5plugin.Blosc.BITSHUFFLE),
                  clevel=1, sensor="IEBCS based event-simulator"):
        """ Save the events into a .hdf5 file with the structure as specified by David Joseph and used by the Cognitive Systems Group of the University of Tübingen.
            Args:
                filename: path of the file
                bias: Bias values (simulator) to save in the file. (th_pos, th_neg, th_n, lat, tau, jit, bgn, refp)
                width: width of the sensor
                height: height of the sensor
                chunk_size: number of events per chunk, default 10 million
                compression: compression method, default Blosc with zstd and bitshuffle
                clevel: compression level, default 1
                sensor: name of the sensor, default "IEBCS based event-simulator"
        """
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
            dset_x.resize((self.i,))
            dset_y.resize((self.i,))
            dset_p.resize((self.i,))
            dset_t.resize((self.i,))
            dset_x[:] = self.get_x()
            dset_y[:] = self.get_y()
            dset_p[:] = self.get_p()
            dset_t[:] = self.get_ts()

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