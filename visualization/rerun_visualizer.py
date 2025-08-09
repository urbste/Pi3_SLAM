"""
Pi3SLAM Real-time Visualization with Viser.
Interactive 3D visualization for SLAM trajectory, point clouds, and camera tracking.
"""

import numpy as np
import time
import multiprocessing as mp
import threading
from typing import Optional, Dict, Any, Tuple, Set
import cv2

try:
    import viser
    import viser.transforms as vtf
    VISER_AVAILABLE = True
except ImportError:
    VISER_AVAILABLE = False
    print("⚠️  Viser not available, falling back to simple visualization")


def render_keypoints_on_image(image: np.ndarray, keypoints: np.ndarray,
                              point_radius: int = 4, line_thickness: int = 2,
                              color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    Efficiently render keypoints on an image with customizable appearance.

    Args:
        image: Input image (H, W, 3) in uint8 format
        keypoints: Keypoint coordinates (N, 2) in pixel coordinates
        point_radius: Radius of keypoint circles
        line_thickness: Thickness of keypoint circles
        color: RGB color tuple for keypoints

    Returns:
        Image with keypoints rendered
    """
    if image is None or keypoints is None or len(keypoints) == 0:
        return image

    rendered_image = image.copy()
    if len(keypoints.shape) == 3:
        keypoints = keypoints.squeeze(0)
    keypoints = keypoints.astype(np.int32)

    height, width = image.shape[:2]
    for kp in keypoints:
        x, y = kp[0], kp[1]
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(rendered_image, (x, y), point_radius, color, line_thickness)

    return rendered_image


class Pi3SLAMViserVisualizer:
    """Real-time interactive SLAM visualization using Viser."""

    def __init__(self, port: int = 8080, max_points: int = 200000):
        self.port = port
        self.max_points = max_points
        self.server = None
        self.running = False

        # Visualization state
        self.total_points = 0
        self.total_cameras = 0
        self.current_chunk = 0

        # GUI controls
        self.gui_controls = {}
        self.point_size = 0.01
        self.camera_size = 0.05
        self.show_trajectory = True
        self.show_cameras = True
        self.show_point_cloud = True

        # Connected clients (for camera control)
        self.clients: Set["viser.ClientHandle"] = set()

        # Follow camera state
        self.auto_follow: bool = False
        self.follow_distance: float = 0.6
        self.follow_height: float = 0.2
        self._latest_cam_pos: Optional[np.ndarray] = None
        self._latest_cam_forward: Optional[np.ndarray] = None

    def start_visualization_server(self, data_queue: mp.Queue):
        """Start the Viser visualization server and data processing."""
        try:
            self.server = viser.ViserServer(port=self.port, verbose=False)
            self.running = True

            print("🎯 Pi3SLAM Viser Visualizer started!")
            print(f"🌐 Open browser to: http://localhost:{self.port}")
            print(f"📊 Max points: {self.max_points:,}")

            self._setup_scene_and_controls()

            data_thread = threading.Thread(
                target=self._process_slam_data,
                args=(data_queue,),
                daemon=True,
            )
            data_thread.start()

            while self.running:
                time.sleep(0.1)

        except Exception as e:
            print(f"❌ Failed to start Viser visualization: {e}")
        finally:
            self.running = False
            if self.server:
                print("🛑 Viser server stopped")

    def _setup_scene_and_controls(self):
        """Setup the 3D scene and GUI controls."""
        self.server.gui.add_markdown(
            "# 🎯 Pi3SLAM Real-time Visualization\n"
            "Interactive 3D SLAM trajectory and point cloud viewer"
        )

        with self.server.gui.add_folder("🎛️ Visualization Controls"):
            self.gui_controls["show_point_cloud"] = self.server.gui.add_checkbox(
                "Show Point Cloud", initial_value=True
            )
            self.gui_controls["show_trajectory"] = self.server.gui.add_checkbox(
                "Show Camera Trajectory", initial_value=True
            )
            self.gui_controls["show_cameras"] = self.server.gui.add_checkbox(
                "Show Camera Poses", initial_value=True
            )
            self.gui_controls["point_size"] = self.server.gui.add_slider(
                "Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.01
            )
            self.gui_controls["camera_size"] = self.server.gui.add_slider(
                "Camera Size", min=0.01, max=0.2, step=0.01, initial_value=0.05
            )

        with self.server.gui.add_folder("📊 SLAM Statistics"):
            self.gui_controls["stats_points"] = self.server.gui.add_text(
                "Total Points", initial_value="0", disabled=True
            )
            self.gui_controls["stats_cameras"] = self.server.gui.add_text(
                "Camera Poses", initial_value="0", disabled=True
            )
            self.gui_controls["stats_chunks"] = self.server.gui.add_text(
                "Chunks Processed", initial_value="0", disabled=True
            )

        with self.server.gui.add_folder("📷 Camera Controls"):
            reset_button = self.server.gui.add_button("Reset View")
            follow_button = self.server.gui.add_button("Follow Latest Camera")
            follow_toggle = self.server.gui.add_checkbox("Auto Follow", initial_value=False)
            self.gui_controls["follow_toggle"] = follow_toggle
            self.gui_controls["follow_distance"] = self.server.gui.add_slider(
                "Follow Distance (m)", min=0.1, max=2.0, step=0.05, initial_value=self.follow_distance
            )
            self.gui_controls["follow_height"] = self.server.gui.add_slider(
                "Follow Height (m)", min=0.0, max=1.0, step=0.05, initial_value=self.follow_height
            )

            @reset_button.on_click
            def _(_: bool) -> None:
                self._reset_camera_view()

            @follow_button.on_click
            def _(_: bool) -> None:
                # One-shot follow to latest camera pose
                self._follow_latest_camera()

            @follow_toggle.on_update
            def _(_: bool) -> None:
                # Continuous follow toggle
                self.auto_follow = follow_toggle.value
                self.follow_distance = float(self.gui_controls["follow_distance"].value)
                self.follow_height = float(self.gui_controls["follow_height"].value)

        @self.server.on_client_connect
        def _on_connect(client: "viser.ClientHandle") -> None:
            try:
                self.clients.add(client)
                client.camera.position = (5.0, 5.0, 3.0)
                client.camera.look_at = (0.0, 0.0, 0.0)
                client.camera.wxyz = (1.0, 0.0, 0.0, 0.0)
            except Exception:
                pass

        @self.server.on_client_disconnect
        def _on_disconnect(client: "viser.ClientHandle") -> None:
            try:
                self.clients.discard(client)
            except Exception:
                pass

    def _process_slam_data(self, data_queue: mp.Queue):
        """Process incoming SLAM data and update visualization."""
        last_update = 0.0
        update_interval = 0.1

        while self.running:
            try:
                slam_data = data_queue.get(timeout=1.0)
                if slam_data is None:
                    break

                current_time = time.time()
                if current_time - last_update < update_interval:
                    continue
                last_update = current_time

                self._update_visualization(slam_data)

            except mp.queues.Empty:
                continue
            except Exception as e:
                print(f"⚠️  Visualization update error: {e}")
                continue

    def _update_visualization(self, slam_data: Dict[str, Any]):
        """Update the 3D visualization with new SLAM data."""
        points = slam_data.get("points", np.array([]))
        colors = slam_data.get("colors", np.array([]))
        camera_positions = slam_data.get("camera_positions", np.array([]))
        camera_orientations = slam_data.get("camera_orientations", np.array([]))
        chunk_poses = slam_data.get("chunk_end_poses", np.array([]))

        self.show_point_cloud = self.gui_controls["show_point_cloud"].value
        self.show_trajectory = self.gui_controls["show_trajectory"].value
        self.show_cameras = self.gui_controls["show_cameras"].value
        self.point_size = self.gui_controls["point_size"].value
        self.camera_size = self.gui_controls["camera_size"].value

        if self.show_point_cloud and len(points) > 0:
            self._update_point_cloud(points, colors)
        if self.show_trajectory and len(camera_positions) > 0:
            self._update_camera_trajectory(camera_positions)
        if self.show_cameras and len(camera_positions) > 0:
            self._update_camera_poses(camera_positions, camera_orientations)
        if len(chunk_poses) > 1:
            self._update_chunk_waypoints(chunk_poses)

        self._update_statistics(slam_data)

        # Update internal latest camera state for follow camera
        if len(camera_positions) > 0:
            self._latest_cam_pos = camera_positions[-1]
            if len(camera_orientations) > 0:
                R = camera_orientations[-1]
                # Forward vector assumed to be +Z axis of camera rotation
                self._latest_cam_forward = R[:, 2]
            else:
                self._latest_cam_forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Apply auto-follow if enabled
        if self.auto_follow and self._latest_cam_pos is not None and self._latest_cam_forward is not None:
            try:
                self.follow_distance = float(self.gui_controls["follow_distance"].value)
                self.follow_height = float(self.gui_controls["follow_height"].value)
            except Exception:
                pass
            self._apply_follow_camera(self._latest_cam_pos, self._latest_cam_forward)

    def _update_point_cloud(self, points: np.ndarray, colors: np.ndarray):
        if len(points) == 0:
            return

        if len(points) > self.max_points:
            step = max(1, len(points) // self.max_points)
            indices = np.arange(0, len(points), step)[: self.max_points]
            display_points = points[indices]
            display_colors = colors[indices]
        else:
            display_points = points
            display_colors = colors

        if display_colors.size > 0 and display_colors.max() > 1.0:
            display_colors = display_colors.astype(np.float32) / 255.0

        self.server.scene.add_point_cloud(
            name="slam_trajectory",
            points=display_points.astype(np.float32),
            colors=display_colors.astype(np.float32),
            point_size=self.point_size,
        )
        self.total_points = len(display_points)

    def _update_camera_trajectory(self, camera_positions: np.ndarray):
        if len(camera_positions) < 2:
            return

        traj_colors = np.full((len(camera_positions), 3), [1.0, 0.2, 0.2], dtype=np.float32)
        line_points: list = []
        line_colors: list = []

        for i in range(len(camera_positions) - 1):
            line_points.extend([camera_positions[i], camera_positions[i + 1]])
            line_colors.extend([traj_colors[i], traj_colors[i + 1]])

        if line_points:
            line_points = np.array(line_points, dtype=np.float32)
            line_colors = np.array(line_colors, dtype=np.float32)
            self.server.scene.add_line_segments(
                name="camera_trajectory",
                points=line_points.reshape(-1, 2, 3),
                colors=line_colors.reshape(-1, 2, 3),
                line_width=3.0,
            )

    def _update_camera_poses(self, camera_positions: np.ndarray, camera_orientations: np.ndarray):
        for i in range(self.total_cameras):
            try:
                self.server.scene.remove(f"camera_{i}")
            except Exception:
                pass

        max_cameras_shown = min(20, len(camera_positions))
        start_idx = max(0, len(camera_positions) - max_cameras_shown)

        for i in range(start_idx, len(camera_positions)):
            position = camera_positions[i]
            if len(camera_orientations) > i:
                rotation_matrix = camera_orientations[i]
                rotation = vtf.SO3.from_matrix(rotation_matrix)
                self.server.scene.add_camera_frustum(
                    name=f"camera_{i}",
                    fov=60.0,
                    aspect=16 / 9,
                    scale=self.camera_size,
                    wxyz=rotation.wxyz,
                    position=position.astype(np.float32),
                    color=[1.0, 0.8, 0.0] if i == len(camera_positions) - 1 else [0.8, 0.2, 0.2],
                )
            else:
                self.server.scene.add_icosphere(
                    name=f"camera_{i}",
                    radius=self.camera_size * 0.5,
                    position=position.astype(np.float32),
                    color=[1.0, 0.8, 0.0] if i == len(camera_positions) - 1 else [0.8, 0.2, 0.2],
                )

        self.total_cameras = len(camera_positions)

    def _update_chunk_waypoints(self, chunk_poses: np.ndarray):
        for i, pose in enumerate(chunk_poses):
            self.server.scene.add_icosphere(
                name=f"chunk_waypoint_{i}",
                radius=self.camera_size * 0.3,
                position=pose.astype(np.float32),
                color=[0.2, 0.8, 0.2],
            )

        if len(chunk_poses) > 1:
            chunk_line_points = []
            for i in range(len(chunk_poses) - 1):
                chunk_line_points.append([chunk_poses[i], chunk_poses[i + 1]])

            if chunk_line_points:
                chunk_line_points = np.array(chunk_line_points, dtype=np.float32)
                chunk_colors = np.full((len(chunk_line_points), 2, 3), [0.2, 0.8, 0.2], dtype=np.float32)
                self.server.scene.add_line_segments(
                    name="chunk_boundaries",
                    points=chunk_line_points,
                    colors=chunk_colors,
                    line_width=2.0,
                )

    def _update_statistics(self, slam_data: Dict[str, Any]):
        try:
            self.gui_controls["stats_points"].value = f"{self.total_points:,}"
            self.gui_controls["stats_cameras"].value = f"{self.total_cameras}"
            frame_info = slam_data.get("frame_info", "")
            if "Chunk:" in frame_info:
                chunk_num = frame_info.split("Chunk:")[1].split(",")[0].strip()
                self.gui_controls["stats_chunks"].value = chunk_num
        except Exception:
            pass

    def _reset_camera_view(self):
        for client in list(self.clients):
            try:
                client.camera.position = (5.0, 5.0, 3.0)
                client.camera.look_at = (0.0, 0.0, 0.0)
                client.camera.wxyz = (1.0, 0.0, 0.0, 0.0)
            except Exception:
                continue

    def _follow_latest_camera(self):
        if self.total_cameras > 0:
            # If we have the latest camera pose, set view behind it
            if self._latest_cam_pos is not None and self._latest_cam_forward is not None:
                self._apply_follow_camera(self._latest_cam_pos, self._latest_cam_forward)
            else:
                # Fallback: reset
                self._reset_camera_view()

    def _apply_follow_camera(self, cam_pos: np.ndarray, cam_forward: np.ndarray) -> None:
        """Position client cameras to follow the latest SLAM camera.
        Places the viewer a fixed distance behind the camera and a bit above, looking at the camera.
        """
        # Normalize forward vector
        fwd = cam_forward.astype(np.float64)
        norm = np.linalg.norm(fwd) + 1e-9
        fwd = fwd / norm
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # Compute a follow position behind the camera and elevated
        follow_pos = cam_pos.astype(np.float64) - fwd * float(self.follow_distance) + up * float(self.follow_height)
        target = cam_pos.astype(np.float64)

        pos_tuple = (float(follow_pos[0]), float(follow_pos[1]), float(follow_pos[2]))
        tgt_tuple = (float(target[0]), float(target[1]), float(target[2]))

        for client in list(self.clients):
            try:
                client.camera.position = pos_tuple
                client.camera.look_at = tgt_tuple
            except Exception:
                continue

    def stop(self):
        self.running = False


def start_viser_visualization_process(data_queue: mp.Queue, port: int = 8080, max_points: int = 200000):
    """Start the Viser visualization in a separate process."""
    if not VISER_AVAILABLE:
        print("❌ Viser not available, cannot start visualization")
        return
    visualizer = Pi3SLAMViserVisualizer(port=port, max_points=max_points)
    visualizer.start_visualization_server(data_queue)


def visualization_process(queue: mp.Queue, rerun_port: int, max_points: int = 200000, update_interval: float = 0.1):
    """Legacy wrapper mapping old Rerun interface to Viser visualization."""
    if VISER_AVAILABLE:
        start_viser_visualization_process(queue, port=rerun_port, max_points=max_points)
    else:
        print("🎥 Visualization process started (console mode)")
        while True:
            try:
                data = queue.get(timeout=1.0)
                if data is None:
                    break
                points = data.get("points", np.array([]))
                cameras = data.get("camera_positions", np.array([]))
                frame_info = data.get("frame_info", "")
                if len(points) > 0 or len(cameras) > 0:
                    print(f"📊 {frame_info} | Points: {len(points):,} | Cameras: {len(cameras)}")
            except mp.queues.Empty:
                continue
            except Exception as e:
                print(f"⚠️  Visualization error: {e}")
                continue
        print("🎥 Visualization process stopped")


def draw_keypoints_on_image(image: np.ndarray, keypoints: np.ndarray, radius: int = 3, thickness: int = 2) -> np.ndarray:
    """Legacy compatibility function for drawing keypoints."""
    return render_keypoints_on_image(image, keypoints, radius, thickness, (0, 0, 255))