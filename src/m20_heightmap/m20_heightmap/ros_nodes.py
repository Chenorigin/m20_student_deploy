"""GOS mapper and AOS observation adapter. Neither node publishes joint commands."""
import json
import copy
import multiprocessing
import os
import secrets
import threading
import time
from collections import deque
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from m20_heightmap_msgs.msg import TerrainObservation, Proprioception, PolicyObservation
from .runtime import load_config, Pipeline
from .geometry import Pose
from .contract import ObservationPacker, validate_model_contract, GRID
from .receiver import receive_terrain
from .commission import geometry_hash, loaded_geometry_commissioned
from .terrain_worker import initialize_worker, estimate_packed_patch, pack_job_inputs


def stamp(msg):
    return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9


def stamp_message(msg, when):
    total = round(when*1e9)
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(total, 1000000000)


def sensor_qos(depth=1):
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=depth,
                      reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)


def q_multiply(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2, w1*w2-x1*x2-y1*y2-z1*z2])


class BaseNode(Node):
    def __init__(self, name):
        super().__init__(name)
        path = self.declare_parameter("config", "").value
        if not path:
            raise ValueError("Pass --ros-args -p config:=/absolute/path/config.json")
        self.cfg = load_config(path)
        self.config_path=Path(path).resolve()
        self.state_lock = threading.RLock()
        self.maintenance_group = MutuallyExclusiveCallbackGroup()
        self.counters, self.last_error = {}, None
        self.status_pub = self.create_publisher(String, self.cfg["topics"]["status"]+"/"+name, 1)
        self.create_timer(1., self.report, callback_group=self.maintenance_group)

    def now(self):
        return self.get_clock().now().nanoseconds*1e-9

    def reject(self, reason):
        with self.state_lock:
            self.counters[reason] = self.counters.get(reason, 0)+1
            self.last_error = reason

    def report(self):
        msg = String()
        with self.state_lock:
            counters = dict(self.counters)
            error = self.last_error
        msg.data = json.dumps({"node": self.get_name(), "rejections": counters,
                               "last_error": error, **self.status()}, allow_nan=False)
        self.status_pub.publish(msg)
        if error:
            self.get_logger().warning(error)

    def status(self):
        return {}


class GosNode(BaseNode):
    def __init__(self):
        super().__init__("m20_heightmap_gos")
        self.pipeline = Pipeline(self.cfg)
        self.session, self.sequence = secrets.randbits(64), 0
        self.map_generation = 0
        self.pose_group = MutuallyExclusiveCallbackGroup()
        self.job_group = self.maintenance_group
        self.pending, self.last_pose_receive = None, None
        self.last_cloud_receive = None
        self.last_publish = None
        self.elapsed_ms = []
        self.job_elapsed_ms, self.result_end_age_ms, self.cloud_header_age_ms = [], [], []
        self.pose_input_age_ms, self.query_ms, self.query_pose_age_ms = [], [], []
        self.pose_input_gap_ms, self.previous_pose_stamp = [], None
        self.pre_publish_pose_age_ms, self.publish_return_pose_age_ms = [], []
        self.query_retries = 0
        self.last_query_valid_cells = None
        self.jobs_submitted, self.jobs_completed, self.cloud_callbacks = 0, 0, 0
        self.future_started, self.future_source_end = None, None
        self.latest_fit_patch = None
        self.trace_query_sources = os.environ.get('M20_TERRAIN_QUERY_TRACE', '0') == '1'
        try:
            self.geometry_id=geometry_hash(self.config_path.parent.parent,self.config_path)
        except OSError:
            self.geometry_id=""
        self.commissioned=False
        self.refresh_commission()
        self.create_timer(1.,self.refresh_commission, callback_group=self.job_group)
        if self.get_parameter('use_sim_time').value:
            raise ValueError('Live terrain process requires the physical system clock')
        worker_cpus = tuple(int(value) for value in os.environ.get(
            'M20_TERRAIN_WORKER_CPUS', os.environ.get('M20_TERRAIN_WORKER_CPU', '6')).split(','))
        self.worker = ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker, initargs=(worker_cpus,))
        self.future = None
        self.future_session = None
        self.worker_fault = None
        self.job_wakeup = self.create_guard_condition(self.job_tick, callback_group=self.job_group)
        self.publish_ms = []
        topics = self.cfg["topics"]
        self.output = self.create_publisher(TerrainObservation, topics["terrain"], sensor_qos())
        cloud_qos=sensor_qos();cloud_qos.reliability=ReliabilityPolicy.RELIABLE
        self.create_subscription(PointCloud2, self.cfg["runtime"]["mapper_cloud_topic"], self.cloud, cloud_qos,
                                 callback_group=self.job_group)
        if self.cfg["pose_mode"] == "odometry":
            self.create_subscription(Odometry, topics["odom"], self.odom, sensor_qos(1),
                                     callback_group=self.pose_group)
        else:
            self.create_subscription(Imu, topics["imu"], self.imu, sensor_qos(32),
                                     callback_group=self.pose_group)
        self.create_timer(1/self.cfg["runtime"]["publish_hz"], self.job_tick,
                          callback_group=self.job_group)
        self.create_timer(1/self.cfg["runtime"]["publish_hz"], self.tick)

    def refresh_commission(self):
        try:
            root=self.config_path.parent.parent
            # Metadata identifies loaded geometry, never a newly edited disk version.
            # Changes require restarting LIO, pose adapter and mapper together.
            self.commissioned=loaded_geometry_commissioned(
                root,self.config_path,self.cfg['profile'],self.geometry_id)
        except OSError:self.commissioned=False

    def covariance(self, covariance, orientation_only=False):
        floors = np.square(self.cfg["runtime"]["pose_std"])
        size = 3 if orientation_only else 6
        cov = np.asarray(covariance).reshape(size, size)
        if not np.isfinite(cov).all() or np.any(np.diag(cov) < 0) or not np.allclose(cov, cov.T, atol=1e-6):
            raise ValueError("Invalid pose covariance")
        if np.linalg.eigvalsh(cov)[0] < -1e-6:
            raise ValueError("Non-positive-semidefinite pose covariance")
        # Spectral bound preserves correlations and is invariant to covariance axes.
        if orientation_only:
            floors[3:] = np.maximum(floors[3:], np.linalg.eigvalsh(cov)[-1])
        else:
            floors[:3] = np.maximum(floors[:3], np.linalg.eigvalsh(cov[:3, :3])[-1])
            floors[3:] = np.maximum(floors[3:], np.linalg.eigvalsh(cov[3:, 3:])[-1])
            # Account conservatively for discarded position-orientation cross correlations.
            cross = np.linalg.norm(cov[:3, 3:], ord=2)
            floors += cross
        return floors

    def push_pose(self, pose):
        now = self.now()
        if now-pose.stamp > self.cfg["runtime"]["input_timeout_s"] or pose.stamp-now > self.cfg["runtime"]["future_tolerance_s"]:
            raise ValueError("Pose timestamp stale/future")
        with self.state_lock:
            try:
                self.pipeline.add_pose(pose)
            except ValueError:
                previous_session = self.session
                self.session, self.sequence = secrets.randbits(64), 0
                self.map_generation += 1
                self.pending = None
                self.latest_fit_patch = None
                self.get_logger().error("Terrain pose epoch reset: "+json.dumps({
                    "previous_session": previous_session, "source_session": self.session,
                    "discontinuity": self.pipeline.last_pose_discontinuity}, allow_nan=False))
                raise
            self.last_pose_receive = time.monotonic()
            if self.previous_pose_stamp is not None:
                self.pose_input_gap_ms.append((pose.stamp-self.previous_pose_stamp)*1000)
                self.pose_input_gap_ms = self.pose_input_gap_ms[-256:]
            self.previous_pose_stamp = pose.stamp
            self.pose_input_age_ms.append((now-pose.stamp)*1000)
            self.pose_input_age_ms = self.pose_input_age_ms[-256:]

    def odom(self, msg):
        try:
            if msg.header.frame_id != self.cfg["frames"]["world"] or msg.child_frame_id != self.cfg["frames"]["base"]:
                raise ValueError("Odometry frame mismatch: need world -> trained base_link")
            p, q, v, w = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear, msg.twist.twist.angular
            self.push_pose(Pose(stamp(msg), [p.x, p.y, p.z], [q.x, q.y, q.z, q.w],
                                self.covariance(msg.pose.covariance), np.linalg.norm([v.x, v.y, v.z]), np.linalg.norm([w.x, w.y, w.z])))
        except ValueError as e:
            self.reject(str(e))

    def imu(self, msg):
        try:
            if msg.header.frame_id != self.cfg["frames"]["imu"] or msg.orientation_covariance[0] < 0:
                raise ValueError("IMU frame/orientation unavailable; raw acceleration is not attitude")
            q, w = msg.orientation, msg.angular_velocity
            inverse = np.asarray(self.cfg["imu_to_base_xyzw"])*[-1, -1, -1, 1]
            qb = q_multiply([q.x, q.y, q.z, q.w], inverse)
            self.push_pose(Pose(stamp(msg), [0., 0., 0.], qb, self.covariance(msg.orientation_covariance, True),
                                0., np.linalg.norm([w.x, w.y, w.z])))
        except ValueError as e:
            self.reject(str(e))

    def cloud(self, msg):
        # Latest-only queue: ROS callbacks never run point processing or grow a backlog.
        with self.state_lock:
            self.pending = msg
            self.last_cloud_receive = time.monotonic()
            self.cloud_callbacks += 1
            self.cloud_header_age_ms.append((self.now()-stamp(msg))*1000)
            self.cloud_header_age_ms = self.cloud_header_age_ms[-256:]
        self.job_wakeup.trigger()

    def destroy_node(self):
        self.worker.shutdown(wait=True)
        return super().destroy_node()

    def job_tick(self):
        # This group serializes cloud preparation and job ownership only. It must
        # never hold state_lock while decoding/copying a scan or waiting for work.
        now = self.now()
        if self.worker_fault is not None:
            return
        if self.future is not None and self.future.done():
            try:
                patch, stats, scan, compute_ms = self.future.result()
                with self.state_lock:
                    if self.future_session != self.session:
                        raise ValueError("Discarded scan from previous pose epoch")
                    mapper = self.pipeline.mapper
                    if not mapper.retain or mapper.profile == "v0":
                        mapper.patches.clear()
                    mapper.patches.append(patch)
                    mapper.last_stamp, mapper.stats = scan.stamp, stats
                    self.map_generation += 1
                    self.pipeline.last_scan = scan
                    self.latest_fit_patch = patch
                    self.jobs_completed += 1
                    self.job_elapsed_ms.append((time.monotonic()-self.future_started)*1000)
                    self.result_end_age_ms.append((self.now()-self.future_source_end)*1000)
                    self.job_elapsed_ms = self.job_elapsed_ms[-256:]
                    self.result_end_age_ms = self.result_end_age_ms[-256:]
                    self.elapsed_ms.append(compute_ms)
                    self.elapsed_ms = self.elapsed_ms[-256:]
            except (ValueError, TypeError) as e:
                self.reject(str(e))
            except Exception as e:
                with self.state_lock:
                    self.worker_fault = type(e).__name__+": "+str(e)
                self.reject("Terrain worker failed: "+self.worker_fault)
            finally:
                self.future = None
        if self.worker_fault is not None:
            return
        with self.state_lock:
            msg = self.pending
            poses = tuple(self.pipeline.poses.poses)
            session, last_stamp = self.session, self.pipeline.mapper.last_stamp
        if msg is not None and self.future is None:
            try:
                if not poses:
                    raise ValueError("No poses for cloud")
                times_end = stamp(msg)
                if self.cfg["runtime"]["point_time_field"]:
                    # Wait for trailing pose samples when scan offsets are positive.
                    from .cloud import read_cloud
                    _, offsets = read_cloud(msg, self.cfg["runtime"]["max_cloud_points"], self.cfg["runtime"]["max_cloud_bytes"],
                                            self.cfg["runtime"]["point_time_field"], self.cfg["runtime"]["point_time_scale_s"],
                                            self.cfg["runtime"].get("point_time_mode", "relative"))
                    if len(offsets):
                        times_end += max(0., float(np.max(offsets)))
                if times_end > poses[-1].stamp and now-times_end <= self.cfg["runtime"]["cloud_input_timeout_s"]:
                    msg = None  # Keep pending, but continue publishing the current map.
                if msg is not None:
                    packed = pack_job_inputs(msg, poses)
                    with self.state_lock:
                        if session != self.session:
                            raise ValueError("Discarded prepared scan from previous pose epoch")
                        self.pending = None
                        self.future_session = session
                        self.future_started, self.future_source_end = time.monotonic(), times_end
                        self.jobs_submitted += 1
                    self.future = self.worker.submit(estimate_packed_patch, self.cfg, packed, stamp(msg),
                                                     self.now(), last_stamp)
                    # Reclaim a completed result immediately instead of waiting
                    # for the next 20ms timer. This callback never fits points.
                    self.future.add_done_callback(lambda _: self.job_wakeup.trigger())
            except (ValueError, TypeError) as e:
                with self.state_lock:
                    if self.pending is msg:
                        self.pending = None
                self.reject(str(e))
            except Exception as e:
                with self.state_lock:
                    self.worker_fault = type(e).__name__+": "+str(e)
                self.reject("Terrain worker failed: "+self.worker_fault)

    def tick(self):
        # snapshot() prunes its mapper. Query a private deque so neither pruning
        # nor NumPy reprojection blocks pose reception or loses a new job result.
        try:
            query_started = time.monotonic()
            with self.state_lock:
                if self.worker_fault is not None:
                    raise ValueError("Terrain worker failed: "+self.worker_fault)
                query = copy.copy(self.pipeline)
                query.poses = copy.copy(self.pipeline.poses)
                query.poses.poses = deque(self.pipeline.poses.poses)
                query.mapper = copy.copy(self.pipeline.mapper)
                query.mapper.trace_sources = self.trace_query_sources
                query.mapper.patches = deque(self.pipeline.mapper.patches,
                                            maxlen=self.pipeline.mapper.cfg.max_packets)
                session, generation = self.session, self.map_generation
                pose_receive, cloud_receive = self.last_pose_receive, self.last_cloud_receive
            pose, (channels, var, age) = query.snapshot(self.now())
            if self.now()-pose.stamp > self.cfg["runtime"]["max_publish_query_age_s"]:
                # A pose may arrive during reprojection. Retry once with that
                # actual newer pose; never relabel old geometry with wall time.
                with self.state_lock:
                    if session != self.session:
                        raise ValueError("Discarded query from previous pose epoch")
                    newer = tuple(self.pipeline.poses.poses)
                    if newer and newer[-1].stamp > pose.stamp:
                        query.poses.poses = deque(newer)
                        pose_receive = self.last_pose_receive
                        self.query_retries += 1
                        retry = True
                    else:
                        retry = False
                if retry:
                    # Keep the captured map and cloud receipt: a new cloud
                    # callback cannot freshen this query's old support.
                    pose, (channels, var, age) = query.snapshot(self.now())
            query_ms = (time.monotonic()-query_started)*1000
            msg = TerrainObservation()
            stamp_message(msg, pose.stamp)
            msg.header.frame_id = self.cfg["frames"]["output"]
            msg.schema_version, msg.profile = 2, self.cfg["profile"]
            msg.session_id = session
            msg.motion_compensated = self.cfg["pose_mode"] == "odometry"
            msg.geometry_hash,msg.operator_commissioned=self.geometry_id,self.commissioned
            msg.height, msg.validity, msg.confidence = [row.tolist() for row in channels]
            msg.variance, msg.age = var.tolist(), age.tolist()
            with self.state_lock:
                if session != self.session:
                    raise ValueError("Discarded query from previous pose epoch")
                if self.worker_fault is not None:
                    raise ValueError("Terrain worker failed: "+self.worker_fault)
                now, mono = self.now(), time.monotonic()
                elapsed = now-pose.stamp
                self.query_ms.append(query_ms)
                self.query_ms = self.query_ms[-256:]
                if np.isfinite(elapsed):
                    self.query_pose_age_ms.append(elapsed*1000)
                    self.query_pose_age_ms = self.query_pose_age_ms[-256:]
                if not np.isfinite(elapsed) or not -self.cfg["runtime"]["future_tolerance_s"] <= elapsed <= self.cfg["runtime"]["max_publish_query_age_s"]:
                    raise ValueError("Terrain query pose expired before publish")
                if pose_receive is None or mono-pose_receive > self.cfg["runtime"]["input_timeout_s"]:
                    raise ValueError("Pose receive watchdog expired")
                # Use the receipts that belong to this snapshot. A newer cloud
                # callback cannot make an old map's watchdog look fresh.
                if cloud_receive is None or mono-cloud_receive > query.mapper.cfg.max_map_age_s:
                    msg.height, msg.validity, msg.confidence = [[0.]*187 for _ in range(3)]
                    msg.variance, msg.age = [0.]*187, [-1.]*187
                msg.sequence = self.sequence
                msg.operator_commissioned = self.commissioned
                publish_started = time.monotonic()
                self.output.publish(msg)
                self.sequence += 1
                self.last_publish = time.monotonic()
                self.pre_publish_pose_age_ms.append(elapsed*1000)
                self.pre_publish_pose_age_ms = self.pre_publish_pose_age_ms[-256:]
                self.publish_return_pose_age_ms.append((self.now()-pose.stamp)*1000)
                self.publish_return_pose_age_ms = self.publish_return_pose_age_ms[-256:]
                valid = np.asarray(msg.validity) == 1
                self.last_query_valid_cells = {
                    "total": int(valid.sum()),
                    "front": int(valid[GRID[:, 0] > 1e-7].sum()),
                    "rear": int(valid[GRID[:, 0] < -1e-7].sum())}
                self.publish_ms.append((self.last_publish-publish_started)*1000)
                self.publish_ms = self.publish_ms[-256:]
                self.last_error = None
                if generation == self.map_generation:
                    self.pipeline.mapper.patches = query.mapper.patches
            if self.trace_query_sources:
                sources = query.mapper.query_sources
                cells = np.flatnonzero(np.asarray(msg.validity) == 1)
                self.get_logger().info('TERRAIN_QUERY '+json.dumps({
                    'schema': 'm20_query_sources_v1', 'source_session': str(session),
                    'sequence': int(msg.sequence), 'geometry_hash': self.geometry_id,
                    'map_generation': generation, 'query_pose_stamp_s': pose.stamp,
                    'actual_wall_ns': time.time_ns(), 'actual_monotonic_ns': time.monotonic_ns(),
                    'theoretical_timestamp_ns': None, 'nominal_period_s': 1/self.cfg['runtime']['publish_hz'],
                    'timestamp_basis': 'support/source doubles unchanged; wall/monotonic after publish; async query',
                    'cells': [[int(i), float(sources['scan_stamp_s'][i]), int(sources['source_cell'][i]),
                               float(sources['oldest_support_s'][i]), float(sources['newest_support_s'][i])]
                              for i in cells]}, allow_nan=False, separators=(',', ':')))
        except ValueError as e:
            self.reject(str(e))

    def status(self):
        with self.state_lock:
            patch = self.latest_fit_patch
            timing = {"compute_p95_ms": list(self.elapsed_ms),
                      "job_wall_p95_ms": list(self.job_elapsed_ms),
                      "result_end_age_p95_ms": list(self.result_end_age_ms),
                      "cloud_header_age_p95_ms": list(self.cloud_header_age_ms),
                      "pose_input_age_p95_ms": list(self.pose_input_age_ms),
                      "pose_input_gap_p95_ms": list(self.pose_input_gap_ms),
                      "query_wall_p95_ms": list(self.query_ms),
                      "query_pose_age_attempt_p95_ms": list(self.query_pose_age_ms),
                      "pre_publish_query_pose_age_p95_ms": list(self.pre_publish_pose_age_ms),
                      "publish_return_query_pose_age_p95_ms": list(self.publish_return_pose_age_ms)}
            publish_ms = list(self.publish_ms)
            result = {"published": self.sequence,
                      "profile": self.cfg["profile"],
                      "terrain_schema_version": 2,
                      "geometry_hash": self.geometry_id,
                      "operator_commissioned": self.commissioned,
                      "source_session": self.session,
                      "published_at_s": self.now(),
                      "last_publish_age_s": None if self.last_publish is None else time.monotonic()-self.last_publish,
                      "worker_fault": self.worker_fault,
                      "jobs_submitted": self.jobs_submitted, "jobs_completed": self.jobs_completed,
                      "cloud_callbacks": self.cloud_callbacks,
                      "query_retries": self.query_retries,
                      "pose_input_gap_max_ms": max(self.pose_input_gap_ms) if self.pose_input_gap_ms else None,
                      "query_valid_cells": self.last_query_valid_cells,
                      "max_publish_query_age_s": self.cfg["runtime"]["max_publish_query_age_s"],
                      "mapping_region": self.cfg["runtime"]["mapping_region"],
                      "mapper_cloud_topic": self.cfg["runtime"]["mapper_cloud_topic"],
                      "fit_anchor_mode": self.cfg["runtime"]["fit_anchor_mode"],
                      "output_mask_cells": int(self.pipeline.output_mask.sum()),
                      "confidence_step_ambiguity_weight": self.pipeline.mapper.cfg.confidence_step_ambiguity_weight,
                      "last_pose_discontinuity": self.pipeline.last_pose_discontinuity,
                      "max_alignment_age_s": self.cfg["aos"]["max_alignment_age_s"],
                      "map": {**self.pipeline.mapper.stats, "packets": len(self.pipeline.mapper.patches)},
                      "pose_mode": self.cfg["pose_mode"]}
        fit_support = None if patch is None else {
            'diagnostic_only': True, 'anchor_stamp': patch.anchor_stamp,
            'fit_valid': patch.valid.astype(int).tolist(), 'blocked': patch.blocked.astype(int).tolist(),
            'oldest_support_age_s': (self.now()-patch.cell_stamps).tolist(),
            'newest_support_age_s': (self.now()-patch.cell_ends).tolist()}
        result.update({key: float(np.percentile(values, 95)) if values else None
                       for key, values in timing.items()})
        result.update(latest_fit_support=fit_support,
                      dds_publish_max_ms=max(publish_ms) if publish_ms else None)
        return result


class AosNode(BaseNode):
    def __init__(self):
        super().__init__("m20_heightmap_aos")
        self.packer = ObservationPacker(self.cfg["profile"], self.cfg["history_steps"])
        sidecar = self.cfg["aos"]["model_contract"]
        if not sidecar:
            raise ValueError("Set aos.model_contract to the final model's observation_contract.json")
        validate_model_contract(sidecar, self.packer.spec)
        self.terrain = None
        self.terrain_received = None
        self.terrain_watermark = None
        self.retired_terrain_sessions = set()
        self.session = None
        self.retired_proprio_sessions = set()
        self.published = 0
        self.output = self.create_publisher(PolicyObservation, self.cfg["topics"]["policy"], sensor_qos())
        self.create_subscription(TerrainObservation, self.cfg["topics"]["terrain"], self.on_terrain, sensor_qos())
        self.create_subscription(Proprioception, self.cfg["topics"]["proprio"], self.on_proprio, sensor_qos(4))

    def on_terrain(self, msg):
        try:
            if self.terrain_watermark is not None:
                last_session, last_sequence = self.terrain_watermark
                if msg.session_id in self.retired_terrain_sessions:
                    raise ValueError("Retired terrain session")
                if msg.session_id == last_session and msg.sequence <= last_sequence:
                    raise ValueError("Repeated/reordered terrain sequence")
            receive_terrain(msg, self.now(), self.cfg)
            if self.terrain_watermark is not None and msg.session_id != self.terrain_watermark[0]:
                if len(self.retired_terrain_sessions) >= 128:
                    raise ValueError("Too many terrain restarts; restart adapter explicitly")
                self.retired_terrain_sessions.add(self.terrain_watermark[0])
                self.packer.reset(clear_watermark=False)
            self.terrain_watermark = (msg.session_id, msg.sequence)
            self.terrain, self.terrain_received = msg, time.monotonic()
        except ValueError as e:
            self.terrain = None
            self.packer.reset(clear_watermark=False)
            self.reject(str(e))

    def on_proprio(self, msg):
        try:
            when, now = stamp(msg), self.now()
            if now-when > self.cfg["aos"]["max_proprio_age_s"] or when-now > self.cfg["runtime"]["future_tolerance_s"]:
                raise ValueError("Proprio timestamp stale/future")
            if msg.header.frame_id != self.cfg["frames"]["base"]:
                raise ValueError("Proprio frame mismatch")
            if self.session != msg.session_id:
                if msg.session_id in self.retired_proprio_sessions or len(self.retired_proprio_sessions) >= 128:
                    raise ValueError("Retired proprio session or too many restarts")
                if self.session is not None:
                    self.retired_proprio_sessions.add(self.session)
                self.packer.reset()
                self.session = msg.session_id
            if self.terrain is None or time.monotonic()-self.terrain_received > self.cfg["aos"]["max_alignment_age_s"]:
                raise ValueError("No fresh terrain received")
            receive_terrain(self.terrain, now, self.cfg)
            channels = receive_terrain(self.terrain, when, self.cfg)
            obs = self.packer.pack(msg.values, channels, int(msg.step), when)
            out = PolicyObservation()
            out.header = msg.header
            out.session_id, out.step = msg.session_id, msg.step
            out.terrain_session_id, out.terrain_sequence = self.terrain.session_id, self.terrain.sequence
            out.profile, out.history_steps = self.cfg["profile"], self.cfg["history_steps"]
            out.values = obs.tolist()
            self.output.publish(out)
            self.published += 1
            self.last_error = None
        except ValueError as e:
            self.packer.reset(clear_watermark=False)
            self.reject(str(e))

    def status(self):
        return {"published": self.published, "input_dim": self.packer.spec["input_dim"],
                "terrain_receive_age_s": None if self.terrain_received is None else time.monotonic()-self.terrain_received}


def run(node_class, args=None):
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = node_class()
        if isinstance(node, GosNode):
            executor = MultiThreadedExecutor(num_threads=3)
            executor.add_node(node)
            executor.spin()
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop callbacks before shutting down the worker or destroying publishers.
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def gos_main():
    run(GosNode)


def aos_main():
    run(AosNode)
