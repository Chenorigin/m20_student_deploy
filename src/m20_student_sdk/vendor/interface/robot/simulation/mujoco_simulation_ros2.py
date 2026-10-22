"""
 * @file mujoco_simulation.py
 * @brief simulation in mujoco
 * @author Bo (Percy) Peng
 * @version 1.0
 * @date 2025-11-05
 *
 * @copyright Copyright (c) 2025  DeepRobotics
"""

import os
import time
import socket
import struct
import threading
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np
import mujoco
import mujoco.viewer

try:
    import cv2
except ImportError:
    cv2 = None

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from drdds.msg import ImuData, JointsData, JointsDataCmd, MetaType, ImuDataValue, JointsDataValue, JointData, JointDataCmd
from std_msgs.msg import Float32MultiArray, MultiArrayDimension



MODEL_NAME = "M20"
# Get the directory of the current Python file
CURRENT_DIR = Path(__file__).resolve().parent

# Define the XML path relative to the Python file
XML_PATH = CURRENT_DIR / ".." / ".." / ".." / "M20_description" / "m20_mjcf" / "mjcf" / "M20_stair.xml"

# Convert to absolute path as string
XML_PATH = str(XML_PATH.resolve())
USE_VIEWER = os.getenv("M20_MUJOCO_VIEWER", "1") == "1"
USE_DEPTH_VIEWER = os.getenv(
    "M20_DEPTH_VIEWER", os.getenv("M20_MUJOCO_VIEWER", "1")
) == "1"
SIM_DEBUG = os.getenv("M20_SIM_DEBUG", "0") == "1"
DT = 0.001
RENDER_INTERVAL = 50

# Perceptive-policy lidar settings. These values intentionally mirror
# DeeproboticsM20PerceptiveMoEStudentV7EnvCfg in the training repository.
LIDAR_CHANNELS = 8
LIDAR_VERTICAL_FOV_DEG = (-28.0, 10.0)
LIDAR_HORIZONTAL_FOV_DEG = (-60.0, 60.0)
LIDAR_HORIZONTAL_SAMPLES = 25  # ceil((60 - -60) / 5) + 1 (both endpoints)
LIDAR_MAX_DISTANCE = 4.0
LIDAR_PERIOD_STEPS = 100       # 10 Hz at DT=0.001 s
LIDAR_HISTORY_LENGTH = 5
LIDAR_STATUS_DIM = 2           # valid_fraction, normalized_age
LIDAR_STATUS_MAX_AGE = 0.5
LIDAR_SCAN_DIM = 2 * LIDAR_CHANNELS * LIDAR_HORIZONTAL_SAMPLES
LIDAR_FRAME_DIM = LIDAR_SCAN_DIM + LIDAR_STATUS_DIM
LIDAR_OBSERVATION_DIM = LIDAR_HISTORY_LENGTH * LIDAR_FRAME_DIM
LIDAR_ADD_TRAINING_NOISE = os.getenv("M20_LIDAR_NOISE", "0") == "1"

# Depth Student V2 camera settings.  The pinhole rays, camera pose, 10 Hz
# cadence, three-frame history, normalization, and chassis mask mirror
# DeeproboticsM20PerceptiveMoEDepthStudentV2EnvCfg exactly.
DEPTH_WIDTH = 64
DEPTH_HEIGHT = 48
DEPTH_FOCAL_LENGTH = 24.0
DEPTH_HORIZONTAL_APERTURE = 45.5503
DEPTH_VERTICAL_APERTURE = 26.6068
DEPTH_CAMERA_OFFSET = np.array([0.23, 0.0, 0.24], dtype=np.float64)
DEPTH_CAMERA_PITCH_DEG = 35.0
DEPTH_MAX_DISTANCE = 4.0
DEPTH_PERIOD_STEPS = 100       # 10 Hz at DT=0.001 s
DEPTH_HISTORY_LENGTH = 3
DEPTH_OCCLUSION_ROWS = 6
DEPTH_OCCLUSION_TOP_WIDTH = 16
DEPTH_OCCLUSION_BOTTOM_WIDTH = 32
DEPTH_FRAME_DIM = DEPTH_HEIGHT * DEPTH_WIDTH
DEPTH_OBSERVATION_DIM = DEPTH_HISTORY_LENGTH * DEPTH_FRAME_DIM + 2
DEPTH_WINDOW_NAME = "M20 Depth (0-4m)"
DEPTH_WINDOW_SCALE = 5

# V11 continuous-stair actor feature.  These rays mirror the 21x15,
# yaw-aligned Isaac Lab v10_pgtt_scanner: local x is [-0.75, 1.25], local y
# is [-0.70, 0.70], and every ray starts 20 m above the base and points down.
# Legacy policies remain positive-only unless their model manifest explicitly
# selects the V11-FD signed-edge contract.
V11_POSITIVE_FEATURE_CONTRACT = "v11_positive_risers_v1"
V11_SIGNED_FEATURE_CONTRACT = "v11_signed_edges_v2"
V11_SIGNED_REAR_AXLE_FEATURE_CONTRACT = "v11_signed_edges_rear_axle_v3"
V11_FEATURE_CONTRACT = os.environ.get(
    "M20_V11_FEATURE_CONTRACT", V11_POSITIVE_FEATURE_CONTRACT,
)
if V11_FEATURE_CONTRACT not in {
    V11_POSITIVE_FEATURE_CONTRACT, V11_SIGNED_FEATURE_CONTRACT,
    V11_SIGNED_REAR_AXLE_FEATURE_CONTRACT,
}:
    raise ValueError(f"unsupported M20_V11_FEATURE_CONTRACT={V11_FEATURE_CONTRACT!r}")
V11_SIGNED_EDGES = V11_FEATURE_CONTRACT in {
    V11_SIGNED_FEATURE_CONTRACT, V11_SIGNED_REAR_AXLE_FEATURE_CONTRACT,
}
V11_REAR_AXLE_DISTANCE = V11_FEATURE_CONTRACT == V11_SIGNED_REAR_AXLE_FEATURE_CONTRACT
V11_GRID_X = np.linspace(1.25, -0.75, 21, dtype=np.float64)
V11_GRID_Y = np.linspace(0.70, -0.70, 15, dtype=np.float64)
V11_RAY_HEIGHT = 20.0
V11_PERIOD_STEPS = 20          # 50 Hz, matching V11 control period
V11_MIN_STEP_HEIGHT = 0.048    # 0.05 m minus V11 height_valid_tolerance
V11_MAX_STEP_HEIGHT = 0.232
V11_CORRIDOR_HALF_WIDTH = 0.20
V11_PASSED_DISTANCE = -0.30
V11_MIN_RISER_SEPARATION = 0.15
V11_ANTICIPATION_MIN = 0.02
V11_ANTICIPATION_MAX = 1.20
V11_REAR_AXLE_DISTANCE_OFFSET = 0.3141
V11_CONFIRMATION_FRAMES = 2
V11_HYSTERESIS_FRAMES = 2
V11_DELAY_STEPS = 2
# Match the V11 training-side actor perturbations.  The standard sim2sim
# launcher enables these by default; direct simulator launches remain clean
# unless explicitly requested.  M20_LIDAR_NOISE is retained as a compatible
# umbrella switch for existing test scripts.
V11_ADD_RADAR_NOISE = os.getenv(
    "M20_V11_RADAR_NOISE", os.getenv("M20_LIDAR_NOISE", "0")
) == "1"
V11_HEIGHT_NOISE_STD = float(os.getenv("M20_V11_HEIGHT_NOISE_STD_M", "0.005"))
V11_DISTANCE_NOISE_STD = float(os.getenv("M20_V11_DISTANCE_NOISE_STD_M", "0.010"))
V11_FRAME_HOLD_PROBABILITY = float(os.getenv("M20_V11_FRAME_HOLD_PROBABILITY", "0.02"))
V11_RADAR_NOISE_SEED = int(os.getenv("M20_V11_RADAR_NOISE_SEED", "0"))

if V11_HEIGHT_NOISE_STD < 0.0 or V11_DISTANCE_NOISE_STD < 0.0:
    raise ValueError("V11 radar noise standard deviations must be non-negative")
if not 0.0 <= V11_FRAME_HOLD_PROBABILITY <= 1.0:
    raise ValueError("M20_V11_FRAME_HOLD_PROBABILITY must be in [0, 1]")

# Calibaration parameters (for sim-to-real consistency)
JOINT_DIR = np.array([1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1], dtype=np.float32)
POS_OFFSET_DEG = np.array([-25, 229, 160, 0, 25, -131, -200, 0, -25, -229, -160, 0, 25, 131, 200, 0], dtype=np.float32)
POS_OFFSET_RAD = POS_OFFSET_DEG / 180.0 * np.pi

JOINT_INIT = {
    "M20": np.array([-0.438, -1.16, 2.76, 0,
                     0.438, -1.16, 2.76, 0,
                     -0.438, 1.16, -2.76, 0,
                     0.438, 1.16, -2.76, 0], dtype=np.float32),
}


class MuJoCoSimulationNode(Node):
    def __init__(self,
                 model_key: str = MODEL_NAME,
                 xml_path: str = XML_PATH):

        super().__init__('mujoco_simulation')

        # 加载 MJCF
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Cannot find MJCF: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)

        # 机器人自由度列表
        self.actuator_ids = [a for a in range(self.model.nu)]  # 0..15
        self.dof_num = len(self.actuator_ids)
        assert self.dof_num == 16, "Expected 16 DOF for M20"

        # 初始化站立姿态
        self._set_initial_pose(model_key)

        # 缓存
        self.kp_cmd = np.zeros((self.dof_num, 1), np.float32)
        self.kd_cmd = np.zeros_like(self.kp_cmd)
        self.pos_cmd = np.zeros_like(self.kp_cmd)
        self.vel_cmd = np.zeros_like(self.kp_cmd)
        self.tau_ff = np.zeros_like(self.kp_cmd)
        self.input_tq = np.zeros_like(self.kp_cmd)

        # IMU
        self.last_base_linvel = np.zeros((3, 1), np.float64)
        self.timestamp = 0.0

        self.lidar_directions = self._make_lidar_directions()
        self.lidar_history = []
        self.lidar_episode_scale = np.random.uniform(0.97, 1.03) if LIDAR_ADD_TRAINING_NOISE else 1.0
        self.lidar_episode_bias = np.random.uniform(-0.04, 0.04) if LIDAR_ADD_TRAINING_NOISE else 0.0
        # Only MuJoCo geom group 0 is visible to the ray caster. Robot collision
        # geoms use group 1 and visuals use group 2, matching Isaac Lab's
        # terrain-only ray casting.
        self.lidar_geom_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        if self.base_body_id < 0:
            raise RuntimeError("MuJoCo model has no base_link body")
        self.depth_directions_camera = self._make_depth_camera_directions()
        self.depth_forward_cosines = self.depth_directions_camera[:, 0].copy()
        pitch_radians = np.deg2rad(DEPTH_CAMERA_PITCH_DEG)
        cosine = np.cos(pitch_radians)
        sine = np.sin(pitch_radians)
        self.depth_rotation_body_from_camera = np.array(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=np.float64,
        )
        self.depth_history = []
        self.depth_geom_ids = np.full(DEPTH_FRAME_DIM, -1, dtype=np.int32)
        self.depth_ray_distances = np.full(DEPTH_FRAME_DIM, -1.0, dtype=np.float64)
        self.depth_ray_cutoff = DEPTH_MAX_DISTANCE / self.depth_forward_cosines.min() + 0.01
        self.depth_viewer_enabled = USE_DEPTH_VIEWER
        self.depth_window_initialized = False
        if self.depth_viewer_enabled and cv2 is None:
            self.get_logger().warn(
                "M20_DEPTH_VIEWER requested, but OpenCV is unavailable; continuing without it"
            )
            self.depth_viewer_enabled = False
        if self.depth_viewer_enabled and not (
            os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")
        ):
            self.get_logger().warn(
                "M20_DEPTH_VIEWER requested, but no graphical display is available; "
                "continuing without it"
            )
            self.depth_viewer_enabled = False

        self.get_logger().info(f"[INFO] MuJoCo model loaded, dof = {self.dof_num}")

        # ROS Publishers
        self.imu_pub = self.create_publisher(ImuData, '/IMU_DATA', 200)
        self.joints_pub = self.create_publisher(JointsData, '/JOINTS_DATA', 200)
        self.lidar_pub = self.create_publisher(Float32MultiArray, '/M20_LIDAR_OBS', 10)
        self.depth_pub = self.create_publisher(Float32MultiArray, '/M20_DEPTH_OBS', 10)
        self.v11_pub = self.create_publisher(Float32MultiArray, '/M20_V11_OBS', 10)

        grid_x, grid_y = np.meshgrid(V11_GRID_X, V11_GRID_Y, indexing="ij")
        self.v11_grid_xy = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)
        self.v11_tracked_feature = np.zeros(6, dtype=np.float32)
        self.v11_confirm_count = np.zeros(2, dtype=np.int32)
        self.v11_invalid_count = np.zeros(2, dtype=np.int32)
        self.v11_feature_ring = [np.zeros(6, dtype=np.float32) for _ in range(V11_DELAY_STEPS + 1)]
        self.v11_last_feature = np.zeros(6, dtype=np.float32)
        self.v11_noise_rng = np.random.default_rng(V11_RADAR_NOISE_SEED)
        self.get_logger().info(
            "[INFO] V11 radar noise "
            f"enabled={V11_ADD_RADAR_NOISE}, "
            f"height_std={V11_HEIGHT_NOISE_STD:.4f} m, "
            f"distance_std={V11_DISTANCE_NOISE_STD:.4f} m, "
            f"frame_hold={V11_FRAME_HOLD_PROBABILITY:.3f}, "
            f"seed={V11_RADAR_NOISE_SEED}"
        )

        # ROS Subscriber
        self.cmd_sub = self.create_subscription(
            JointsDataCmd,
            '/JOINTS_CMD',
            self._cmd_callback,
            50
        )

        # 可视化
        self.viewer = None
        if USE_VIEWER:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _set_initial_pose(self, key: str):
        """关节位置设置为与 PyBullet 脚本一致的初始角度"""
        qpos0 = self.data.qpos.copy()
        qpos0[7:7 + self.dof_num] = JOINT_INIT[key]  # ,3-6 basequat，0-2 basepos
        qpos0[:3] = np.array([0, 0, 0.2])
        qpos0[3:7] = np.array([1, 0, 0, 0])
        self.data.qpos[:] = qpos0
        mujoco.mj_forward(self.model, self.data)

    def _cmd_callback(self, msg: JointsDataCmd):
        """Convert received (published) positions/velocities to internal (raw)"""
        if len(msg.data.joints_data) != 16:
            self.get_logger().warn("Received JointsDataCmd with incorrect number of joints")
            return

        pub_pos = np.zeros(self.dof_num, dtype=np.float32)
        pub_vel = np.zeros(self.dof_num, dtype=np.float32)
        for i in range(self.dof_num):
            joint_cmd = msg.data.joints_data[i]
            self.kp_cmd[i] = joint_cmd.kp
            self.kd_cmd[i] = joint_cmd.kd
            pub_pos[i] = joint_cmd.position
            pub_vel[i] = joint_cmd.velocity
            self.tau_ff[i] = joint_cmd.torque  # tau_ff no processing

        # Convert: raw = published * dir + offset_rad
        self.pos_cmd.flat = pub_pos * JOINT_DIR + POS_OFFSET_RAD
        self.vel_cmd.flat = pub_vel * JOINT_DIR

    def start(self):
        # 主模拟循环
        step = 0
        last_time = time.time()
        while rclpy.ok():
            if time.time() - last_time >= DT:
                last_time = time.time()
                step += 1
                # 控制律
                self._apply_joint_torque()
                # 模拟一步
                mujoco.mj_step(self.model, self.data)

                self.timestamp = step * DT

                # 采样 & 发送观测 (every 5 steps for 200 Hz)
                if step % 5 == 0:
                    self._publish_robot_state(step)

                if step % LIDAR_PERIOD_STEPS == 0:
                    self._update_and_publish_lidar()

                if step % DEPTH_PERIOD_STEPS == 0:
                    self._update_and_publish_depth()

                if step % V11_PERIOD_STEPS == 0:
                    self._update_and_publish_v11()

                if SIM_DEBUG and step % 500 == 0:
                    rpy_deg = np.rad2deg(self.quaternion_to_euler(self.data.qpos[3:7]))
                    self.get_logger().info(
                        "[M20_SIM_DEBUG] "
                        f"t={self.timestamp:.2f} "
                        f"base_xyz={self.data.qpos[:3].round(4).tolist()} "
                        f"rpy_deg={rpy_deg.round(2).tolist()}"
                    )

                # 可视化
                if self.viewer and step % RENDER_INTERVAL == 0:
                    self.viewer.sync()

            # Handle ROS callbacks
            rclpy.spin_once(self, timeout_sec=0.0)

    def _apply_joint_torque(self):
        # 当前关节状态
        q = self.data.qpos[7:7 + self.dof_num].reshape(-1, 1)
        dq = self.data.qvel[6:6 + self.dof_num].reshape(-1, 1)
        self.input_tq = (
                self.kp_cmd * (self.pos_cmd - q) +
                self.kd_cmd * (self.vel_cmd - dq) +
                self.tau_ff
        )

        # 写入 control 缓冲区
        self.data.ctrl[:] = self.input_tq.flatten()

    @staticmethod
    def _make_lidar_directions():
        """Reproduce Isaac Lab's LidarPatternCfg ray ordering exactly."""
        vertical = np.linspace(*LIDAR_VERTICAL_FOV_DEG, LIDAR_CHANNELS)
        horizontal = np.linspace(*LIDAR_HORIZONTAL_FOV_DEG, LIDAR_HORIZONTAL_SAMPLES)
        vertical_rad, horizontal_rad = np.meshgrid(
            np.deg2rad(vertical), np.deg2rad(horizontal), indexing="ij"
        )
        directions = np.stack(
            [
                np.cos(vertical_rad) * np.cos(horizontal_rad),
                np.cos(vertical_rad) * np.sin(horizontal_rad),
                np.sin(vertical_rad),
            ],
            axis=-1,
        )
        return directions.reshape(-1, 3)

    def _cast_lidar(self, origin_world, directions_world):
        ranges = np.empty(directions_world.shape[0], dtype=np.float32)
        valid = np.empty(directions_world.shape[0], dtype=bool)
        geom_id = np.array([-1], dtype=np.int32)
        for index, direction in enumerate(directions_world):
            geom_id[0] = -1
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                origin_world,
                direction,
                self.lidar_geom_group,
                1,
                -1,
                geom_id,
            )
            valid[index] = np.isfinite(distance) and distance >= 0.0
            ranges[index] = LIDAR_MAX_DISTANCE if not valid[index] else min(distance, LIDAR_MAX_DISTANCE)
        return ranges, valid

    @staticmethod
    def _make_depth_camera_directions():
        """Build Isaac Lab-compatible pinhole rays in row-major image order."""
        focal_x = DEPTH_WIDTH * DEPTH_FOCAL_LENGTH / DEPTH_HORIZONTAL_APERTURE
        focal_y = DEPTH_HEIGHT * DEPTH_FOCAL_LENGTH / DEPTH_VERTICAL_APERTURE
        center_x = DEPTH_WIDTH / 2.0
        center_y = DEPTH_HEIGHT / 2.0
        pixel_x, pixel_y = np.meshgrid(
            np.arange(DEPTH_WIDTH, dtype=np.float64) + 0.5,
            np.arange(DEPTH_HEIGHT, dtype=np.float64) + 0.5,
            indexing="xy",
        )
        # Isaac Lab converts the conventional camera frame (right, down,
        # forward) to the robotics frame (forward, left, up).
        directions = np.stack(
            [
                np.ones_like(pixel_x),
                -(pixel_x - center_x) / focal_x,
                -(pixel_y - center_y) / focal_y,
            ],
            axis=-1,
        ).reshape(-1, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return directions

    @staticmethod
    def _mask_depth_chassis(depth, valid):
        """Apply the centered V2 chassis mask used during training."""
        first_row = DEPTH_HEIGHT - DEPTH_OCCLUSION_ROWS
        for row_offset in range(DEPTH_OCCLUSION_ROWS):
            alpha = row_offset / max(DEPTH_OCCLUSION_ROWS - 1, 1)
            mask_width = round(
                DEPTH_OCCLUSION_TOP_WIDTH
                + alpha * (DEPTH_OCCLUSION_BOTTOM_WIDTH - DEPTH_OCCLUSION_TOP_WIDTH)
            )
            left = (DEPTH_WIDTH - mask_width) // 2
            row = first_row + row_offset
            depth[row, left:left + mask_width] = DEPTH_MAX_DISTANCE
            valid[row, left:left + mask_width] = False

    def _update_and_publish_depth(self):
        """Publish the depth V2 actor's 9218-D history and status message."""
        rotation_world_from_body = self.data.xmat[self.base_body_id].reshape(3, 3)
        camera_origin = (
            self.data.xpos[self.base_body_id]
            + rotation_world_from_body @ DEPTH_CAMERA_OFFSET
        )
        rotation_world_from_camera = (
            rotation_world_from_body @ self.depth_rotation_body_from_camera
        )
        directions_world = np.ascontiguousarray(
            self.depth_directions_camera @ rotation_world_from_camera.T,
            dtype=np.float64,
        )

        self.depth_geom_ids.fill(-1)
        self.depth_ray_distances.fill(-1.0)
        mujoco.mj_multiRay(
            self.model,
            self.data,
            np.ascontiguousarray(camera_origin, dtype=np.float64),
            directions_world.reshape(-1),
            self.lidar_geom_group,
            1,
            -1,
            self.depth_geom_ids,
            self.depth_ray_distances,
            DEPTH_FRAME_DIM,
            self.depth_ray_cutoff,
        )

        image_plane_depth = self.depth_ray_distances * self.depth_forward_cosines
        valid = (
            np.isfinite(image_plane_depth)
            & (self.depth_ray_distances >= 0.0)
            & (image_plane_depth > 0.0)
            & (image_plane_depth <= DEPTH_MAX_DISTANCE)
        )
        depth = np.where(valid, image_plane_depth, DEPTH_MAX_DISTANCE).reshape(
            DEPTH_HEIGHT, DEPTH_WIDTH
        )
        valid = valid.reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
        self._mask_depth_chassis(depth, valid)
        valid_fraction = float(np.mean(valid))
        self._show_depth_image(depth, valid, valid_fraction)

        normalized_depth = np.clip(
            depth / DEPTH_MAX_DISTANCE * 2.0 - 1.0, -1.0, 1.0
        ).astype(np.float32)
        if not self.depth_history:
            self.depth_history = [normalized_depth.copy() for _ in range(DEPTH_HISTORY_LENGTH)]
        else:
            self.depth_history = self.depth_history[1:] + [normalized_depth]

        normalized_age = 0.0 if valid_fraction > 0.5 else 1.0
        observation = np.concatenate(
            [
                *(frame.reshape(-1) for frame in self.depth_history),
                np.array([valid_fraction, normalized_age], dtype=np.float32),
            ]
        ).astype(np.float32, copy=False)
        if observation.size != DEPTH_OBSERVATION_DIM or not np.all(np.isfinite(observation)):
            raise RuntimeError(
                f"Depth observation is invalid; expected {DEPTH_OBSERVATION_DIM} finite values"
            )
        msg = Float32MultiArray()
        msg.data = observation.tolist()
        self.depth_pub.publish(msg)

        if SIM_DEBUG:
            self.get_logger().info(
                "[M20_SIM_DEPTH] "
                f"valid={valid_fraction:.3f} "
                f"min={depth.min():.3f} max={depth.max():.3f}"
            )

    def _show_depth_image(self, depth, valid, valid_fraction):
        """Show the newest policy input as a compact colorized depth image."""
        if not self.depth_viewer_enabled:
            return
        try:
            # Near terrain is red/yellow, far terrain is blue/purple, and
            # invalid/chassis-mask pixels are black.
            proximity = np.clip(
                (DEPTH_MAX_DISTANCE - depth) / DEPTH_MAX_DISTANCE * 255.0,
                0.0,
                255.0,
            ).astype(np.uint8)
            colorized = cv2.applyColorMap(proximity, cv2.COLORMAP_TURBO)
            colorized[~valid] = 0
            display = cv2.resize(
                colorized,
                (DEPTH_WIDTH * DEPTH_WINDOW_SCALE, DEPTH_HEIGHT * DEPTH_WINDOW_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            label = f"near:red  far:blue  valid:{valid_fraction:.1%}"
            cv2.putText(
                display,
                label,
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if not self.depth_window_initialized:
                cv2.namedWindow(DEPTH_WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(
                    DEPTH_WINDOW_NAME,
                    DEPTH_WIDTH * DEPTH_WINDOW_SCALE,
                    DEPTH_HEIGHT * DEPTH_WINDOW_SCALE,
                )
                cv2.moveWindow(DEPTH_WINDOW_NAME, 20, 20)
                self.depth_window_initialized = True
            cv2.imshow(DEPTH_WINDOW_NAME, display)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("v")):
                self.close_depth_viewer()
        except cv2.error as error:
            self.get_logger().warn(
                f"Disabling depth viewer after OpenCV GUI error: {error}"
            )
            self.close_depth_viewer()

    def close_depth_viewer(self):
        if self.depth_window_initialized and cv2 is not None:
            try:
                cv2.destroyWindow(DEPTH_WINDOW_NAME)
                cv2.waitKey(1)
            except cv2.error:
                pass
        self.depth_window_initialized = False
        self.depth_viewer_enabled = False

    def _update_and_publish_lidar(self):
        """Publish the V7 actor's 2010-D lidar-history observation term."""
        base_position = self.data.qpos[:3].copy()
        quat_wxyz = self.data.qpos[3:7].copy()
        rotation_world_from_body = Rotation.from_quat(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        ).as_matrix()

        front_origin = base_position + rotation_world_from_body @ np.array([0.34, 0.0, -0.01])
        rear_origin = base_position + rotation_world_from_body @ np.array([-0.34, 0.0, -0.01])
        front_directions = self.lidar_directions @ rotation_world_from_body.T
        rear_rotation = rotation_world_from_body @ np.diag([-1.0, -1.0, 1.0])
        rear_directions = self.lidar_directions @ rear_rotation.T
        front_scan, front_valid = self._cast_lidar(front_origin, front_directions)
        rear_scan, rear_valid = self._cast_lidar(rear_origin, rear_directions)
        scan = np.concatenate([front_scan, rear_scan])
        valid = np.concatenate([front_valid, rear_valid])

        if LIDAR_ADD_TRAINING_NOISE:
            scan = scan * self.lidar_episode_scale + self.lidar_episode_bias
            scan += np.random.normal(0.0, 0.035, scan.shape)
            dropped_rays = np.random.random(scan.shape) < 0.02
            scan[dropped_rays] = LIDAR_MAX_DISTANCE
            valid[dropped_rays] = False
            if np.random.random() < 0.04:
                scan.fill(LIDAR_MAX_DISTANCE)
                valid.fill(False)
        scan = np.clip(scan, 0.0, LIDAR_MAX_DISTANCE)
        normalized_scan = scan / LIDAR_MAX_DISTANCE * 2.0 - 1.0
        valid_fraction = float(np.mean(valid))

        if not self.lidar_history:
            normalized_age = 0.0 if valid_fraction > 0.5 else 1.0
            new_frame = np.concatenate(
                [normalized_scan, [valid_fraction, normalized_age]]
            ).astype(np.float32)
            self.lidar_history = [new_frame] * LIDAR_HISTORY_LENGTH
        elif LIDAR_ADD_TRAINING_NOISE and np.random.random() < 0.08:
            previous = self.lidar_history[-1]
            normalized_age = min(
                float(previous[-1]) + (LIDAR_PERIOD_STEPS * DT / LIDAR_STATUS_MAX_AGE),
                1.0,
            )
            new_frame = np.concatenate(
                [previous[:LIDAR_SCAN_DIM], [0.0, normalized_age]]
            ).astype(np.float32)
            self.lidar_history = self.lidar_history[1:] + [new_frame]
        else:
            previous_age = float(self.lidar_history[-1][-1])
            normalized_age = (
                0.0
                if valid_fraction > 0.5
                else min(
                    previous_age + (LIDAR_PERIOD_STEPS * DT / LIDAR_STATUS_MAX_AGE),
                    1.0,
                )
            )
            new_frame = np.concatenate(
                [normalized_scan, [valid_fraction, normalized_age]]
            ).astype(np.float32)
            self.lidar_history = self.lidar_history[1:] + [new_frame]

        observation = np.clip(np.concatenate(self.lidar_history), -1.0, 1.0).astype(np.float32)
        if observation.size != LIDAR_OBSERVATION_DIM:
            raise RuntimeError(
                f"Lidar observation has {observation.size} values; expected {LIDAR_OBSERVATION_DIM}"
            )
        msg = Float32MultiArray()
        msg.data = observation.tolist()
        self.lidar_pub.publish(msg)

    def _update_and_publish_v11(self):
        """Publish V11's delayed, confirmed two-riser [B,H,D] actor feature."""
        rotation_world_from_body = self.data.xmat[self.base_body_id].reshape(3, 3)
        yaw = np.arctan2(rotation_world_from_body[1, 0], rotation_world_from_body[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        rotation_world_from_yaw = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
        base_position = self.data.xpos[self.base_body_id]
        local_xyz = np.column_stack((self.v11_grid_xy, np.zeros(self.v11_grid_xy.shape[0])))
        origins = base_position + local_xyz @ rotation_world_from_yaw.T
        origins[:, 2] = base_position[2] + V11_RAY_HEIGHT
        points = np.full((21, 15, 3), np.nan, dtype=np.float64)
        geom_id = np.array([-1], dtype=np.int32)
        for index, origin in enumerate(origins):
            geom_id[0] = -1
            distance = mujoco.mj_ray(
                self.model, self.data, np.ascontiguousarray(origin), np.array((0.0, 0.0, -1.0)),
                self.lidar_geom_group, 1, -1, geom_id,
            )
            if np.isfinite(distance) and distance >= 0.0:
                row, column = divmod(index, 15)
                points[row, column] = (self.v11_grid_xy[index, 0], self.v11_grid_xy[index, 1], origin[2] - distance)

        detected = self._extract_v11_two_risers(points)
        detected_bits = np.array((detected[0] > 0.5, detected[3] > 0.5), dtype=bool)
        prior_bits = np.array((self.v11_tracked_feature[0] > 0.5, self.v11_tracked_feature[3] > 0.5), dtype=bool)
        self.v11_confirm_count = np.where(detected_bits, self.v11_confirm_count + 1, 0)
        self.v11_invalid_count = np.where(detected_bits, 0, self.v11_invalid_count + 1)
        actor_bits = (detected_bits & (self.v11_confirm_count >= V11_CONFIRMATION_FRAMES)) | (
            prior_bits & (self.v11_invalid_count <= V11_HYSTERESIS_FRAMES)
        )
        for slot, offset in enumerate((0, 3)):
            if detected_bits[slot]:
                self.v11_tracked_feature[offset + 1:offset + 3] = detected[offset + 1:offset + 3]
            self.v11_tracked_feature[offset] = float(actor_bits[slot])
            if not actor_bits[slot]:
                self.v11_tracked_feature[offset + 1:offset + 3] = 0.0

        noisy_feature = self._apply_v11_radar_noise(self.v11_tracked_feature)
        self.v11_feature_ring = self.v11_feature_ring[1:] + [noisy_feature]
        feature = self.v11_feature_ring[0].copy()
        # Training masks H/D after the delayed frame is read.  Noise therefore
        # cannot create geometry when the corresponding B bit is false.
        feature[1:3] *= feature[0]
        feature[4:6] *= feature[3]
        self.v11_last_feature = feature.copy()
        if not np.all(np.isfinite(feature)):
            raise RuntimeError("V11 feature must be finite")
        msg = Float32MultiArray()
        msg.layout.dim = [MultiArrayDimension(
            label=("B1,signedH1,D1,B2,signedH2,D2" if V11_SIGNED_EDGES
                   else "B1,H1,D1,B2,H2,D2"),
            size=6, stride=6,
        )]
        msg.data = feature.tolist()
        self.v11_pub.publish(msg)
        if SIM_DEBUG:
            self.get_logger().info(f"[M20_SIM_V11] feature={np.round(feature, 3).tolist()}")

    def _apply_v11_radar_noise(self, tracked_feature):
        """Apply the bounded V11 training perturbation before the delay ring."""
        noisy = np.asarray(tracked_feature, dtype=np.float32).copy()
        if noisy.shape != (6,) or not np.all(np.isfinite(noisy)):
            raise ValueError("V11 tracked feature must contain six finite values")
        if not V11_ADD_RADAR_NOISE:
            return noisy

        if self.v11_noise_rng.random() < V11_FRAME_HOLD_PROBABILITY:
            noisy = self.v11_last_feature.copy()
        noisy[[1, 4]] += self.v11_noise_rng.normal(0.0, V11_HEIGHT_NOISE_STD, 2)
        noisy[[2, 5]] += self.v11_noise_rng.normal(0.0, V11_DISTANCE_NOISE_STD, 2)
        # Signed-edge contracts retain negative heights. Presence bits and
        # distances are non-negative in both contracts.
        noisy[[0, 2, 3, 5]] = np.maximum(noisy[[0, 2, 3, 5]], 0.0)
        height_min = -V11_MAX_STEP_HEIGHT if V11_SIGNED_EDGES else 0.0
        noisy[[1, 4]] = np.clip(noisy[[1, 4]], height_min, V11_MAX_STEP_HEIGHT)
        return noisy.astype(np.float32, copy=False)

    @staticmethod
    def _extract_v11_two_risers(points):
        """Publish V11's two-riser actor feature for one MuJoCo robot.

        Raw detection still identifies the current and next risers.  When the
        current riser leaves the forward anticipation window, the next riser
        becomes B1/H1/D1, matching the V11 training-side continuous handoff.
        """
        # Convert semantic front-to-rear rows to increasing yaw-local x.
        ascending = points[::-1]
        rows = []
        for index in range(ascending.shape[0] - 1):
            low, upper = ascending[index], ascending[index + 1]
            finite = np.isfinite(low[:, 2]) & np.isfinite(upper[:, 2])
            height = upper[:, 2] - low[:, 2]
            magnitude = np.abs(height)
            x = 0.5 * (low[:, 0] + upper[:, 0])
            y = 0.5 * (low[:, 1] + upper[:, 1])
            sign_allowed = np.ones_like(height, dtype=bool) if V11_SIGNED_EDGES else height > 0.0
            candidate = finite & sign_allowed & (np.abs(y) <= V11_CORRIDOR_HALF_WIDTH) & (
                magnitude >= V11_MIN_STEP_HEIGHT
            ) & (magnitude <= V11_MAX_STEP_HEIGHT)
            if not np.any(candidate):
                continue
            column = np.argmax(np.where(candidate, magnitude, -np.inf))
            if x[column] > V11_PASSED_DISTANCE:
                rows.append((float(x[column]), float(height[column])))
        if not rows:
            return np.zeros(6, dtype=np.float32)
        def actor_distance(row):
            x, height = row
            if V11_REAR_AXLE_DISTANCE and height < 0.0:
                return x + V11_REAR_AXLE_DISTANCE_OFFSET
            return x

        first_index = min(range(len(rows)), key=lambda index: abs(actor_distance(rows[index])))
        first_x, first_height = rows[first_index]
        later = [
            (x, height) for x, height in rows
            if x > first_x + V11_MIN_RISER_SEPARATION and (height > 0.0) == (first_height > 0.0)
        ]
        first_distance = actor_distance((first_x, first_height))
        first_in_window = V11_ANTICIPATION_MIN <= first_distance <= V11_ANTICIPATION_MAX
        if first_in_window:
            if not later:
                return np.array((1.0, first_height, first_distance, 0.0, 0.0, 0.0), dtype=np.float32)
            second_x, second_height = min(later, key=lambda row: row[0])
            second_distance = actor_distance((second_x, second_height))
            if V11_REAR_AXLE_DISTANCE and not (
                V11_ANTICIPATION_MIN <= second_distance <= V11_ANTICIPATION_MAX
            ):
                return np.array((
                    1.0, first_height, first_distance, 0.0, 0.0, 0.0,
                ), dtype=np.float32)
            return np.array((
                1.0, first_height, first_distance,
                1.0, second_height, second_distance,
            ), dtype=np.float32)

        # The closest raw riser has been crossed.  Hand the next visible riser
        # to actor slot 1 so V11 stays in its perceptive stair gait.
        if later:
            second_x, second_height = min(later, key=lambda row: row[0])
            second_distance = actor_distance((second_x, second_height))
            if V11_ANTICIPATION_MIN <= second_distance <= V11_ANTICIPATION_MAX:
                return np.array((
                    1.0, second_height, second_distance, 0.0, 0.0, 0.0,
                ), dtype=np.float32)
        return np.zeros(6, dtype=np.float32)

    # --------------------------------------------------------
    def quaternion_to_euler(self, q):
        """
        Convert a quaternion to Euler angles (roll, pitch, yaw).
        """
        w, x, y, z = q

        # roll (X-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        # pitch (Y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)  # 防止数值漂移导致 |t2|>1
        pitch = np.arcsin(t2)

        # yaw (Z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        return np.array([roll, pitch, yaw], dtype=np.float32)

    # --------------------------------------------------------

    def _publish_robot_state(self, step: int):
        # ----- IMU -----
        q_world = self.data.sensordata[:4]  # quaternion (w, x, y, z) in MuJoCo convention
        rpy_rad = self.quaternion_to_euler(q_world)  # returns [roll, pitch, yaw] in radians

        # Convert to degrees
        rpy_deg = [angle * (180.0 / 3.141592653589793) for angle in rpy_rad]

        body_acc = self.data.sensordata[4:7]
        angvel_b = self.data.sensordata[7:10]  # body frame

        imu_msg = ImuData()
        imu_msg.header = MetaType()
        imu_msg.header.frame_id = 0
        stamp = Time()
        sec = int(self.timestamp)
        nanosec = int((self.timestamp - sec) * 1e9)
        stamp.sec = sec
        stamp.nanosec = nanosec
        imu_msg.header.stamp = stamp
        imu_msg.data = ImuDataValue()
        imu_msg.data.roll = float(rpy_deg[0])
        imu_msg.data.pitch = float(rpy_deg[1])
        imu_msg.data.yaw = float(rpy_deg[2])
        imu_msg.data.omega_x = float(angvel_b[0])
        imu_msg.data.omega_y = float(angvel_b[1])
        imu_msg.data.omega_z = float(angvel_b[2])
        imu_msg.data.acc_x = float(body_acc[0])
        imu_msg.data.acc_y = float(body_acc[1])
        imu_msg.data.acc_z = float(body_acc[2])
        self.imu_pub.publish(imu_msg)

        # ----- 关节 -----
        q = self.data.qpos[7:7 + self.dof_num]
        dq = self.data.qvel[6:6 + self.dof_num]
        tau = self.input_tq.flatten()

        # Convert raw to published: published = (raw - offset_rad) * dir
        pub_pos = (q - POS_OFFSET_RAD) * JOINT_DIR
        pub_vel = dq * JOINT_DIR
        pub_tau = tau * JOINT_DIR  # Torque also needs direction flip
        
        joints_msg = JointsData()
        joints_msg.header = MetaType()
        joints_msg.header.frame_id = 0
        stamp = Time()
        sec = int(self.timestamp)
        nanosec = int((self.timestamp - sec) * 1e9)
        stamp.sec = sec
        stamp.nanosec = nanosec
        joints_msg.header.stamp = stamp
        joints_msg.data = JointsDataValue()
        joints_msg.data.joints_data = [JointData() for _ in range(self.dof_num)]
        for i in range(self.dof_num):
            joint = joints_msg.data.joints_data[i]
            joint.name = [32, 32, 32, 32]  # Dummy name (four spaces)
            joint.data_id = 0  # Dummy
            joint.status_word = 1  # Normal
            joint.position = float(pub_pos[i])
            joint.torque = float(pub_tau[i])
            joint.velocity = float(pub_vel[i])
            joint.motion_temp = 40.0  # Dummy normal temp
            joint.driver_temp = 45.0  # Dummy normal temp
        self.joints_pub.publish(joints_msg)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    rclpy.init()
    sim_node = MuJoCoSimulationNode()
    try:
        sim_node.start()
    except KeyboardInterrupt:
        pass
    except Exception:
        # rl_deploy and the simulator share a shell/process group.  During a
        # normal shutdown the ROS context can become invalid between the loop's
        # rclpy.ok() check and a publish/spin call.  Suppress only that shutdown
        # race; propagate genuine runtime failures while ROS is still active.
        if rclpy.ok():
            raise
    finally:
        sim_node.close_depth_viewer()
        sim_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
