""" Simulator instance for simulating a ball-trajectory using blender

    This file provedes a class to run a simulation on a given blender scene and .yaml config file.
    The configurations match "default.yaml" in the config folder, but can be adapted to other scenes and configurations as well.
    The simulator is tuned to match the configurations in the table-tennis robot lab at the University of Tübingen.

    After the most recent pipeline changes to match the newer version of the IEBCS project, this simulator outptus a .mp4 video which the can be used as imput for the event-simulator
    So the pipeline is split up in two steps and can be parallelized if needed.
"""
import bpy
import bpy_extras
from bpy_extras import anim_utils
import cv2
import numpy as np
import pandas as pd
import time
import sys
import subprocess
sys.path.append('./utils/')
import rotations



class BlenderSimulator:
    """ Class to run a simulation on a given blender scene and .yaml config file. """

    def __init__(self, config):
        self.set_config(config)
        self.ball_coords = []
        self.calculate_fps()

    
    def set_config(self, config):
        """ Set the configuration for the simulator. """
        self.config = config
        
        # paths
        self.dataset_path = config['dataset_path']
        self.tmp_path = config['tmp_path']
        self.output_path = config['output_path']
        self.scene_path = config['scene_path']
        self.coords_path = config['coords_path']
        self.metadata_path = config['metadata_path']
        self.gt_path = config['gt_path']

        # settings
        self.stop_early = config['stop_early']

        # positional settings
        o = config["initial_orientation"]
        self.initial_orientation = rotations.Rotation()
        self.initial_orientation.set_axis(o[0], o[1], o[2])
        self.ball_start = config["ball_start"]
        self.ball_end = config["ball_end"]
        self.ball_scale_start = config["scale_start"]
        self.ball_scale_end = config["scale_end"]

        # spin-settings
        self.spin_axis = config["spin_axis"]
        self.spin = rotations.Rotation()
        self.spin.set_axis_np(np.array(self.spin_axis))
        self.total_rotations = config["total_rotations"]

        # resolution and video settings
        self.resolution_x = config["resolution_x"]
        self.resolution_y = config["resolution_y"]
        self.resolution_precentage = config["resolution_percentage"]
        self.focal_length = config["focal_length"]
        self.pixel_pitch = config["pixel_pitch"]
        self.total_frames = config["total_frames"]
        self.simulation_time = config["simulation_time"]
        self.video_fps = config["video_fps"]
        self.simulation_samples = config["simulation_samples"]
        self.ball_name = config["ball_name"]


    def calculate_fps(self):
        """ Calculate the frames per second for the simulation. 
            This is based on the settings for total frames and simulation time, which are set in the config file.       
        """
        self.video_length = self.simulation_time / 1000000.0  # convert microseconds to seconds
        self.fps = int(self.total_frames / self.video_length)

    
    def init_scene(self):
        """ Initialize the blender scene for the simulation. """
        # load the scene
        bpy.ops.wm.open_mainfile(filepath=self.scene_path)
        self.ball = bpy.data.objects[self.ball_name]
        self.scene = bpy.context.scene

        # set the render settings
        bpy.context.scene.render.resolution_x = self.resolution_x
        bpy.context.scene.render.resolution_y = self.resolution_y
        bpy.context.scene.render.fps = self.fps
        bpy.context.scene.render.image_settings.file_format = 'PNG'

        # set animation settings
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = self.total_frames 

        # set background
        world = bpy.data.worlds.get("World")
        if world.node_tree is None:
            world.use_nodes = True
            world.node_tree = bpy.data.node_groups.new(type="ShaderNodeTree", name="WorldNodeTree")

        bg = world.node_tree.nodes["Background"]
        bg.inputs[0].default_value = (0.1, 0.1, 0.1, 1)  # Set background color to dark gray
        bg.inputs[1].default_value = 0.0  # Set background strength (black)


    def init_camera(self):
        """ Initialize the camera for the simulation. """
        self.cam = bpy.data.objects['Camera']
        self.cam.data.lens = self.focal_length
        self.cam.data.sensor_width = self.pixel_pitch * self.resolution_x * self.resolution_precentage
        self.cam.data.sensor_height = self.pixel_pitch * self.resolution_y * self.resolution_precentage
        self.cam.data.sensor_fit = 'HORIZONTAL'  # Fit the sensor to the horizontal dimension
        bpy.context.scene.eevee.taa_render_samples = self.simulation_samples
        bpy.context.scene.eevee.taa_samples = self.simulation_samples


    def apply_initial_rotation(self):
        """ Apply the initial rotation to the ball. """
        self.ball.rotation_mode = 'AXIS_ANGLE'
        ax = self.initial_orientation.get_axis()
        angle = self.initial_orientation.get_angle()
        angle = angle * np.pi / 180.0  # convert to radians
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
        ad = self.ball.animation_data
        channelbag = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
        #for fcurve in self.ball.animation_data.action.fcurves:
        for fcurve in channelbag.fcurves:
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
            "camera_position_world_x": [self.cam.location[0]],
            "camera_position_world_y": [self.cam.location[1]],
            "camera_position_world_z": [self.cam.location[2]],
            "camera_rotation_world_x": [self.cam.rotation_euler[0]],
            "camera_rotation_world_y": [self.cam.rotation_euler[1]],
            "camera_rotation_world_z": [self.cam.rotation_euler[2]],
            "ball_scale_start": [self.ball_scale_start],
            "ball_scale_end": [self.ball_scale_end],
            "total_frames": [self.total_frames],
            "total_rotations": [self.total_rotations],
            "video_length": [self.video_length],
            "fps": [self.fps],
        }
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(self.metadata_path, index=False)

        # Save ball coordinates per frame as CSV
        coords_df = pd.DataFrame(self.ball_coords, columns=["frame", "screen_position"])
        # Split screen_position tuple into two columns
        coords_df[["screen_x", "screen_y"]] = pd.DataFrame(coords_df["screen_position"].tolist(), index=coords_df.index)
        coords_df = coords_df.drop(columns=["screen_position"])
        coords_df.to_csv(self.coords_path, index=False)


    def simulate(self):
        """ Runs the blender simulation part
        
            Setup needs to be completed for this to work
        """
        start_ts = time.time()

        for frame in range(self.scene.frame_start, self.scene.frame_end + 1):
            self.scene.frame_set(frame)
            out_png = self.tmp_path + f"frame_{frame:06d}.png"
            self.scene.render.filepath = out_png

            bpy.ops.render.render(write_still=True)
            self.update_ground_truth(frame)
            
            if self.stop_early and frame >= 40:
                break

        out_avi = self.output_path + "frames.avi"
        in_pattern = self.tmp_path + "frame_%06d.png"

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(self.fps),
            "-i", in_pattern,
            "-c:v", "mjpeg",
            "-q:v", "3",
            out_avi,
        ]
        
        subprocess.run(cmd, check=True)

        self.save_ground_truth()
        duration = time.time() - start_ts
        print(f"Simulation completed in {duration:.2f} seconds.")


    def run_simulation(self):
        """ Run the full simulation pipeline """
        self.init_scene()
        self.init_camera()
        self.apply_initial_rotation()
        self.generate_spin_keyframes()
        self.generate_scale_keyframes()
        self.generate_position_keyframes()
        self.simulate()


if __name__ == "__main__":
    # load config
    import yaml
    with open("../config/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    simulator = BlenderSimulator(config)
    simulator.run_simulation()