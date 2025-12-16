import time
import mujoco
import mujoco.viewer
import numpy as np
import cv2

class MujocoSim:
    def __init__(self, model_source):
        """
        Initializes and loads the simulation.

        Args:
            model_source (str): A file path to a .xml file OR a string
                                containing the XML model definition.
        """
        try:
            if model_source.strip().startswith("<mujoco>"):
                self.model = mujoco.MjModel.from_xml_string(model_source)
            else:
                self.model = mujoco.MjModel.from_xml_path(model_source)
            
            self.data = mujoco.MjData(self.model)
            self.viewer = None
            print(f"loaded model: {self.model.nq} DoFs")

        except Exception as e:
            print(f"Failed to initialize simulation: {e}")
            raise

    def get_nx(self):
        """
        Returns the total dimension of the state vector (nx = nq + nv).
        """
        return self.model.nq + self.model.nv

    def launch_viewer(self, width=1280, height=720):
        """
        Launches the passive viewer for this simulation instance.
        
        Args:
            width (int): Desired viewer window width (may be limited by MuJoCo)
            height (int): Desired viewer window height (may be limited by MuJoCo)
        """
        if self.viewer is None or not self.viewer.is_running():
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # Note: The actual window size is controlled by MuJoCo and may not match exactly
            print(f"Viewer launched")
        else:
            print("Viewer is already running")
        return self.viewer

    def step(self, ctrl=None):
        """
        Performs one simulation step.
        Accepts ctrl as either row or column vector.
        """
        if ctrl is not None:
            ctrl = np.asarray(ctrl).reshape(-1)
            assert ctrl.shape == (self.model.nu,), \
                f"ctrl shape mismatch. Expected {(self.model.nu,)}, got {ctrl.shape}"
            self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data)

        if self.is_viewer_running():
            self.viewer.sync()

    def set_state_vector(self, state):
        """
        Sets full state from a vector: state = [qpos; qvel]
        Accepts both (n,) and (n,1) shapes.
        """
        state = np.asarray(state).reshape(-1)

        nq = self.model.nq
        nv = self.model.nv
        assert state.shape == (nq + nv,), \
            f"Expected state vector of shape {(nq+nv,)}, got {state.shape}"

        self.data.qpos[:] = state[:nq]
        self.data.qvel[:] = state[nq:]

        mujoco.mj_forward(self.model, self.data)

    def set_state(self, qpos, qvel):
        """
        Sets qpos and qvel. Accepts row or column vectors.
        """
        qpos = np.asarray(qpos).reshape(-1)
        qvel = np.asarray(qvel).reshape(-1)

        state = np.concatenate([qpos, qvel])
        self.set_state_vector(state)

    def set_control(self, ctrl):
        """Set actuator controls without advancing simulation."""
        assert ctrl.shape == (self.model.nu,), f"ctrl shape mismatch. Expected {(self.model.nu,)}, got {ctrl.shape}"
        self.data.ctrl[:] = ctrl

    def get_state(self):
        state = np.concatenate([self.data.qpos, self.data.qvel])
        return state[:, None]      # column vector for matlab interpretation

    def get_time(self):
        """Returns the current simulation time."""
        return self.data.time

    def get_timestep(self):
        """Returns the simulation timestep (dt)."""
        return self.model.opt.timestep

    def is_viewer_running(self):
        """Checks if the viewer is currently open and running."""
        return self.viewer is not None and self.viewer.is_running()
    
    def close_viewer(self):
        """Closes the viewer window cleanly."""
        if self.is_viewer_running():
            self.viewer.close()
            self.viewer = None
            print("Viewer closed")

    def start_recording(self, filename="simulation.mp4", fps=30, width=640, height=480):
        """
        Starts recording frames from the viewer.
        
        Args:
            filename (str): Output video filename
            fps (int): Frames per second for the output video
            width (int): Frame width in pixels
            height (int): Frame height in pixels
        """
        
        if not self.is_viewer_running():
            raise RuntimeError("Viewer must be running to start recording.")
        
        self.recording = True
        self.video_filename = filename
        self.video_fps = fps
        self.video_width = width
        self.video_height = height
        self.frames = []
        
        # Initialize a persistent renderer for fast frame capture
        self.recorder_renderer = mujoco.Renderer(self.model, height=height, width=width)
        
    def capture_frame(self, sim_time=None):
        """
        Captures the current viewer frame if recording is active.
        
        Args:
            sim_time (float, optional): Simulation time to display on frame
        """
        if hasattr(self, 'recording') and self.recording:
            if self.is_viewer_running():
                # Use the persistent renderer
                self.recorder_renderer.update_scene(self.data, camera=self.viewer.cam)
                pixels = self.recorder_renderer.render()
                
                # Add time annotation if provided
                if sim_time is not None:
                    pixels = self._add_time_annotation(pixels, sim_time)
                
                self.frames.append(pixels)

    def _add_time_annotation(self, frame, sim_time):
        """
        Adds simulation time text overlay to frame.
        """
        annotated = frame.copy()
        time_text = f"t = {sim_time:.3f} s"
        
        # Set up text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 1
        font_color = getattr(self, 'font_color', (255, 255, 255))
        
        (text_width, text_height), baseline = cv2.getTextSize(
            time_text, font, font_scale, thickness
        )
        
        # position of text
        x = (self.video_width - text_width) // 2
        y = text_height + 20
        
        # Add text in RGB
        cv2.putText(
            annotated,
            time_text,
            (x, y),
            font,
            font_scale,
            font_color,
            thickness,
            cv2.LINE_AA
        )
        
        return annotated

    def stop_recording(self):
        """Stops recording and saves the video file."""
        
        if not hasattr(self, 'recording') or not self.recording:
            print("No active recording to stop.")
            return
        
        self.recording = False
        
        # Clean up the persistent renderer
        if hasattr(self, 'recorder_renderer'):
            self.recorder_renderer.close()
            del self.recorder_renderer
        
        if len(self.frames) == 0:
            print("No frames captured.")
            return
                
        # Define the codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            self.video_filename, 
            fourcc, 
            self.video_fps,
            (self.video_width, self.video_height)
        )
        
        for frame in self.frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            out.write(frame_bgr)
        
        out.release()
        self.frames = []  # Clear frames for memory
        print(f"Video saved to {self.video_filename}")

    def set_camera(self, azimuth=None, elevation=None, distance=None, lookat=None):
        """
        Sets the viewer camera parameters.

        Args:
            azimuth (float, optional): Rotation around vertical axis (degrees).
            elevation (float, optional): Vertical angle (degrees).
            distance (float, optional): Distance from the lookat point.
            lookat (array-like, optional): 3D point [x, y, z] the camera looks at.
        """
        if not self.is_viewer_running():
            raise RuntimeError("Viewer is not running. Call launch_viewer() first.")
        
        cam = self.viewer.cam  # shortcut
        
        if azimuth is not None:
            cam.azimuth = azimuth
        if elevation is not None:
            cam.elevation = elevation
        if distance is not None:
            cam.distance = distance
        if lookat is not None:
            lookat = np.asarray(lookat)
            assert lookat.shape == (3,), "lookat must be a 3-element array [x,y,z]"
            cam.lookat[:] = lookat



