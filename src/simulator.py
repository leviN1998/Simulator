""" Simulator instance, for simulating a ball-trajectory

    This file runs a simulation with a given config (see in configs/simulator/default.yaml)
    The simulation is set up in a way that every parameter is set by the config file.
    Ball positions, spin, rps, video length etc. need to be set in the config file, or be overwrittenwhen launching a simulaiton
"""
import bpy
import bpy_extras
import cv2
import time
import os
import sys
import numpy as np
import pandas as pd
sys.path.append("src/utils")
sys.path.append("src/IEBCS")
sys.path.append("../src/IEBCS") # debug
sys.path.append("../src/utils")       # debug
import rotations
import eventIO
from dvs_sensor import *
from dvs_sensor_blender import Blender_DvsSensor


class Simulator:
    """ Simulator instance

        This class is used to run a simulation with a given config file.
    """

    def __init__(self, config, logger, simulation_nr=0, pid=0):
        """ Initialize the simulator instance

            Args:
                config (dict): Configuration dictionary with simulation parameters
                logger (Logger): Logger instance for logging messages
                simulation_nr (int): Simulation number for naming output files
                pid (int): Process ID for logging purposes
        """
        self.logger = logger
        self.set_config(config)
        self.simulation_nr = simulation_nr
        self.pid = pid
        self.num_string = str(simulation_nr).zfill(5)
        self.tmp_path = self.dataset_path + f"tmp/pid_{self.pid}/image_tmp.png"
        self.output_name = self.dataset_path + f"data/{self.num_string}/{self.num_string}_"
        self.scene_path = self.dataset_path + f"config/scene.blend"
        self.coords_path = self.output_name + "ball_coords.csv"
        self.metadata_path = self.output_name + "metadata.csv"
        self.gt_path = self.output_name + "ground_truth.csv"
        self.ball_coords = []
        self.calculate_fps()

        try:
            os.mkdir(self.dataset_path + "data/" + self.num_string)
        except FileExistsError:
            self.logger.error(f"Directory {self.dataset_path}data/{self.num_string} already exists. Please remove it before running the simulation again.")


    def set_config(self, config):
        """ Set the configuration for the simulation
        """
        self.generate_video = config["generate_video"]
        self.stop_early = config["stop_early"]

        self.dataset_path = config["dataset_path"]

        o = config["initial_orientation"]
        self.initial_orientation = rotations.Rotation()
        self.initial_orientation.set_axis(o[0], o[1], o[2])
        self.spin_axis = config["spin_axis"]
        self.spin = rotations.Rotation()
        self.spin.set_axis_np(np.array(self.spin_axis))
        self.total_rotations = config["total_rotations"]

        self.ball_start = config["ball_start"]
        self.ball_end = config["ball_end"]
        self.ball_scale_start = config["scale_start"]
        self.ball_scale_end = config["scale_end"]

        self.total_frames = config["total_frames"]
        self.simulation_time = config["simulation_time"]
        self.video_fps = config["video_fps"]
        self.simulation_samples = config["simulation_samples"]
        self.ball_name = config["ball_name"]

        self.resolution_x = config["resolution_x"]
        self.resolution_y = config["resolution_y"]
        self.resolution_percentage = config["resolution_percentage"]
        self.focal_length = config["focal_length"]
        self.pixel_pitch = config["pixel_pitch"]

        self.th_pos = config["th_pos"]
        self.th_neg = config["th_neg"]
        self.th_n = config["th_n"]
        self.lat = config["lat"]
        self.tau = config["tau"]
        self.jit = config["jit"]
        self.bgn = config["bgn"]
        self.ref_period = config["ref_period"]


    def calculate_fps(self):
        """ Calculate the frames per second (fps) based on the simulation time and number of frames.
        """
        self.video_length = self.simulation_time / 1000000.0  # convert microseconds to seconds
        self.fps = int(self.total_frames / self.video_length)


    def init_scene(self):
        """ Initialize the Blender scene for the simulation.
        """
        bpy.ops.wm.open_mainfile(filepath=self.scene_path)
        self.ball = bpy.data.objects[self.ball_name]
        self.scene = bpy.context.scene

        self.scene.frame_start = 0
        self.scene.frame_end = self.total_frames
        self.scene.render.fps = self.fps
        self.scene.render.image_settings.file_format = 'PNG'

        # set background
        bpy.data.worlds["World"].use_nodes = True
        bg = bpy.data.worlds["World"].node_tree.nodes["Background"]
        bg.inputs[0].default_value = (0.1, 0.1, 0.1, 1)  # R, G, B, Alpha (black)
        bg.inputs[1].default_value = 0.0


    def init_camera(self):
        """ Set up ev camera """
        cam_pos = bpy.data.objects["Camera"].location
        cam_rot = bpy.data.objects["Camera"].rotation_euler
        self.event_camera = Blender_DvsSensor("Sensor")
        self.event_camera.cam = bpy.data.objects["Camera"]
        self.event_camera.set_sensor(nx=self.resolution_x, ny=self.resolution_y, pp=self.pixel_pitch)
        # print(self.th_pos, self.th_neg, self.th_n, self.lat, self.tau, self.jit, self.bgn, self.ref_period)
        self.event_camera.set_dvs_sensor(th_pos=self.th_pos, th_neg=self.th_neg, th_n=self.th_n, lat=self.lat, tau=self.tau, jit=self.jit, bgn=self.bgn)
        self.event_camera.ref = self.ref_period
        self.event_camera.set_sensor_optics(self.focal_length)
        self.scene.render.resolution_x = self.resolution_x
        self.scene.render.resolution_y = self.resolution_y
        self.scene.render.resolution_percentage = self.resolution_percentage
        self.scene.eevee.taa_render_samples = self.simulation_samples

        self.scene.camera = self.event_camera.cam
        self.event_camera.set_position(cam_pos)
        self.event_camera.set_angle(cam_rot)
        self.event_camera.init_tension()
        self.event_camera.init_bgn_hist(self.dataset_path + "noise/noise_pos_161lux.npy", self.dataset_path + "noise/noise_pos_161lux.npy")


    def apply_initial_rotation(self):
        """
        Apply initial rotation to the ball.
        The initial rotation is defined in the config file.
        """
        self.ball.rotation_mode = 'AXIS_ANGLE'
        ax = self.initial_orientation.get_axis()
        angle = self.initial_orientation.get_angle()
        self.logger.debug(f"Initial rotation axis: {ax} with {angle} deg.")
        angle = angle * np.pi / 180.0 # convert to radians
        self.ball.rotation_axis_angle = (angle, ax[0], ax[1], ax[2])
        # apply initial rotation
        # make it active and the only selected object
        bpy.context.view_layer.objects.active = self.ball
        for o in bpy.context.selected_objects:
            o.select_set(False)
        self.ball.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)


    def generate_spin_keyframes(self):
        """ Generate keyframes for the ball spin
        """
        self.ball.rotation_mode = 'AXIS_ANGLE'
        ax = self.spin.get_axis()
        self.ball.rotation_axis_angle = (0, ax[0], ax[1], ax[2])
        self.ball.keyframe_insert(data_path="rotation_axis_angle", frame=0, index=-1)

        self.ball.rotation_axis_angle = (self.total_rotations * np.pi * 2, ax[0], ax[1], ax[2])
        self.ball.keyframe_insert(data_path="rotation_axis_angle", frame=self.total_frames, index=-1)


    def generate_scale_keyframes(self):
        """ Generate keyframes for the ball scale
        """
        self.ball.scale = (self.ball_scale_start, self.ball_scale_start, self.ball_scale_start)
        self.ball.keyframe_insert(data_path="scale", frame=0)

        self.ball.scale = (self.ball_scale_end, self.ball_scale_end, self.ball_scale_end)
        self.ball.keyframe_insert(data_path="scale", frame=self.total_frames)


    def generate_position_keyframes(self):
        """ Generate keyframes for the ball position

            The ball is moved from the start position to the end position over the total frames.
        """
        self.ball.location = self.ball_start
        self.ball.keyframe_insert(data_path="location", frame=0)

        self.ball.location = self.ball_end
        self.ball.keyframe_insert(data_path="location", frame=self.total_frames)

        # Set interpolation to linear for constant rotation speed
        for fcurve in self.ball.animation_data.action.fcurves:
            for kf in fcurve.keyframe_points:
                kf.interpolation = 'LINEAR'


    def get_screen_positions(self):
        ''' Returns the screen coords of the ball

            This Funciton should return the screen coords of the ball, to use it in the ground truth file
            The network should only get the ball-area as input

            At the moment this only calculates the position in pixels.
            Maybe it would be beneficial to also include the size of the ball ROI
            -> This was buggy in the last implementation so just let it be a parameter for now

        '''
        center = bpy_extras.object_utils.world_to_camera_view(
            scene=self.scene,
            obj=self.scene.camera,
            coord=self.ball.location
        )
        render = self.scene.render
        res_x = render.resolution_x * render.resolution_percentage / 100
        res_y = render.resolution_y * render.resolution_percentage / 100

        pixel_x = center.x * res_x
        pixel_y = (1 - center.y) * res_y

        return pixel_x, pixel_y


    def update_ground_truth(self, frame):
        """ Update the ground truth file with the current frame's data

            This function updates the pixel coords of the ball at the given frame.
        """
        
        self.ball_coords.append((frame, self.get_screen_positions()))


    def save_ground_truth(self):
        """ Save the ground truth data to a file
        """
        # Save ground truth (rotation info) as CSV
        gt = {
            "rotation_x": [self.spin.get_axis()[0]],
            "rotation_y": [self.spin.get_axis()[1]],
            "rotation_z": [self.spin.get_axis()[2]],
            "rotation_omega": [self.spin.get_angle()],
        }
        gt_df = pd.DataFrame(gt)
        gt_df.to_csv(self.gt_path, index=False)
        self.logger.debug(f"Ground truth data saved to {self.gt_path}")

        # Save metadata as CSV
        metadata = {
            "rotation_x": [self.spin.get_axis()[0]],
            "rotation_y": [self.spin.get_axis()[1]],
            "rotation_z": [self.spin.get_axis()[2]],
            "rotation_omega": [self.spin.get_angle()],
            "ball_start_x": [self.ball_start [0]],
            "ball_start_y": [self.ball_start[1]],
            "ball_start_z": [self.ball_start[2]],
            "ball_end_x": [self.ball_end[0]],
            "ball_end_y": [self.ball_end[1]],
            "ball_end_z": [self.ball_end[2]],
            "initial_rot_x": [self.initial_orientation.get_axis()[0]],
            "initial_rot_y": [self.initial_orientation.get_axis()[1]],
            "initial_rot_z": [self.initial_orientation.get_axis()[2]],
            "ball_position_world_x": [self.ball.location[0]],
            "ball_position_world_y": [self.ball.location[1]],
            "ball_position_world_z": [self.ball.location[2]],
            "camera_position_world_x": [self.event_camera.cam.location[0]],
            "camera_position_world_y": [self.event_camera.cam.location[1]],
            "camera_position_world_z": [self.event_camera.cam.location[2]],
            "camera_rotation_world_x": [self.event_camera.cam.rotation_euler[0]],
            "camera_rotation_world_y": [self.event_camera.cam.rotation_euler[1]],
            "camera_rotation_world_z": [self.event_camera.cam.rotation_euler[2]],
            "ball_scale_start": [self.ball_scale_start],
            "ball_scale_end": [self.ball_scale_end],
            "total_frames": [self.total_frames],
            "total_rotations": [self.total_rotations],
            "video_length": [self.video_length],
            "fps": [self.fps],
        }
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(self.metadata_path, index=False)
        self.logger.debug(f"Metadata saved to {self.metadata_path}")

        # Save ball coordinates per frame as CSV
        coords_df = pd.DataFrame(self.ball_coords, columns=["frame", "screen_position"])
        # Split screen_position tuple into two columns
        coords_df[["screen_x", "screen_y"]] = pd.DataFrame(coords_df["screen_position"].tolist(), index=coords_df.index)
        coords_df = coords_df.drop(columns=["screen_position"])
        coords_df.to_csv(self.coords_path, index=False)
        self.logger.debug(f"Ball coordinates saved to {self.coords_path}")


    def redirect_output(self):
        """ Redirect the blender output to a file

            This function redirects the blender output to a file, so that the logs can be saved
            and used for debugging purposes.
        """
        logfile = self.dataset_path + f"tmp/pid_{self.pid}/render.log"
        with open(logfile, 'a') as f:
            f.write("\n\n========== NEW BLENDER OUTPUT ==========\n\n")
            f.write(f"Simulation {self.simulation_nr} started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n\n")
        self.old = os.dup(sys.stdout.fileno())
        sys.stdout.flush()
        os.close(sys.stdout.fileno())
        self.fd = os.open(logfile, os.O_WRONLY | os.O_APPEND)


    def restore_output(self):
        os.close(self.fd)
        os.dup(self.old)
        os.close(self.old)


    def simulate(self):
        """ Runs the blender simulation part

            Setup needs to be done for this to work
        """

        if self.generate_video:
            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
            video = cv2.VideoWriter(self.output_name + "frames.avi", fourcc, self.video_fps, (self.resolution_x, self.resolution_y))

        ev = EventBuffer(0)

        if not self.stop_early:
            self.redirect_output()
        start_ts = time.time()
        end_ts = time.time()
        for frame in range(self.scene.frame_start, self.scene.frame_end+1):
            duration = end_ts - start_ts
            start_ts = time.time()

            if frame % 100 == 0:
                self.logger.progress(f"Simulation {self.simulation_nr}: Rendering frame {frame}/{self.scene.frame_end}  ({duration:.2f} s/frame, {int(duration*self.total_frames)}s total.)")
            self.scene.frame_set(frame)

            self.scene.render.filepath = self.tmp_path
            bpy.ops.render.render(write_still=True)
            img = cv2.imread(self.tmp_path)

            self.logger.debug(f"Ball Location: {self.ball.location}, Frame: {frame}, Image shape: {img.shape}")

            self.update_ground_truth(frame)
            if self.generate_video:
                video.write(img)

            if frame == 0:
                self.event_camera.init_image(img)
            else:
                delta_t = 1000000.0 * (1.0 / self.fps)  # delta t in us (1000000 us = 1 s)
                pk = self.event_camera.update(img, delta_t)
                ev.increase_ev(pk)
            end_ts = time.time()

            if self.stop_early and frame >= 20:
                self.logger.info("Stopping early for debugging purposes.")
                break

        if not self.stop_early:
            self.restore_output()

        if self.generate_video:
            video.release()
            self.logger.debug(f"Video saved to {self.output_name}frames.avi")

        bias = [self.th_pos, self.th_neg, self.th_n, self.lat, self.tau, self.jit, self.bgn, self.ref_period]
        eventIO.save_hdf5(ev, self.output_name + "events.hdf5", bias)

        self.save_ground_truth()


    def run_simulation(self):
        """ Run the simulation.
        
            Execute all setup steps, the run the simualtion
        """
        self.init_scene()
        self.init_camera()
        self.apply_initial_rotation()
        self.generate_spin_keyframes()
        self.generate_scale_keyframes()
        self.generate_position_keyframes()
        self.simulate()
        self.logger.info(f"Simulation {self.simulation_nr} finished. Output saved to {self.output_name}")


if __name__ == "__main__":
    print("test simulator module")

    import yaml
    import logger

    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    logger = logger.Logger(path="/data/lkolmar/datasets/test/tmp/")

    sim = Simulator(config, logger, simulation_nr=0, pid=0)
    sim.run_simulation()
    print("Simulation finished.")

    import eventIO
    import event_representations
    buf = eventIO.load_hdf5("/data/lkolmar/datasets/test/data/00000/00000_events.hdf5")
    print("Duration: ", (buf.get_ts().max() - buf.get_ts().min()), "us")
    evs = eventIO.buffer_to_array(buf)
    out = eventIO.EventBuffer(0)

    width, height = 1280, 720
    x, y = 56, 355
    left = int(max(x - 50, 0))
    right = int(min(x + 50, width))
    top = int(max(y - 50, 0))
    bottom = int(min(y + 50, height))
    mask = (
        (evs["x"] >= left) & (evs["x"] < right) &
        (evs["y"] >= top) & (evs["y"] < bottom)
    )
    events = evs[mask]
    events["x"] -= left
    events["y"] -= top
    out.add_array(events["t"], events["y"], events["x"], events["p"])
    
    sequence = event_representations.create_sequence(eventIO.buffer_to_array(out), time_window=5000, num_bins=10, sensor_size=(100, 100), flip=False, normalize=False)
    print("Sequence shape: ", sequence.shape)
    real_path = f"/home/lkolmar/Documents/metavision/recordings/dataset_full_ball-gun/roi/spike_5_spin_6_rec2_converted.hdf5"
    real_buf = eventIO.load_hdf5(real_path)
    real_sequence = event_representations.create_sequence(eventIO.buffer_to_array(real_buf), time_window=5000, num_bins=10, sensor_size=(100, 100), flip=False, normalize=False)
    print(f"Total events: {np.sum(sequence[1])}, Real events: {np.sum(real_sequence[1])}")
    print(f"Ratio sim/real: {np.sum(sequence[1])/np.sum(real_sequence[1])}")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(20, 5))
    axes[0].imshow(event_representations.get_voxel_grid_as_image(sequence[1]))
    axes[0].set_title("Simulated events")
    axes[1].imshow(event_representations.get_voxel_grid_as_image(real_sequence[1]))
    axes[1].set_title("Real events")

    plt.show()
    fig.savefig("/data/lkolmar/datasets/test/data/00000/plot.png")