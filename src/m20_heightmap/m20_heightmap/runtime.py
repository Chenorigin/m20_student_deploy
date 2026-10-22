"""Shared live/replay path, without ROS imports."""
import json
import math
import re
import time
from pathlib import Path
import numpy as np
from .estimator import ElevationMapper, EstimatorConfig
from .geometry import Pose, PoseBuffer, rotation, to_world, deskew, yaw_rotation
from .cloud import read_cloud, validate_cloud_time
from .contract import contract, finite_vector, GRID
from .visibility import validate_visibility, visible_body, front_map_mask


def load_config(path):
    path = Path(path).expanduser().resolve()
    cfg = json.loads(path.read_text())
    required = {"profile", "history_steps", "calibration_confirmed", "pose_mode", "topics", "frames",
                "lidar_extrinsics", "imu_to_base_xyzw", "self_filter_boxes", "estimator", "runtime", "aos"}
    if not required <= set(cfg) or set(cfg) - required - {'raw_input', 'visibility'}:
        raise ValueError("Config missing/unknown keys: " + str(set(cfg)^required))
    if 'raw_input' in cfg:
        from .native_packet import SOURCE, PROTOCOL
        if cfg['raw_input'] != {'protocol':PROTOCOL,'source':SOURCE}:
            raise ValueError('Unsupported native DDS input contract')
        if 'visibility' not in cfg:
            raise ValueError('Native deployment requires calibrated visibility')
    if 'visibility' in cfg:
        reference=cfg['visibility']
        if set(reference) != {'site_config'} or not isinstance(reference['site_config'],str):
            raise ValueError('Visibility must reference the physical site configuration')
        site=json.loads((path.parent/reference['site_config']).resolve().read_text())
        cfg['visibility']=validate_visibility(site['lidar_visibility'])
    contract(cfg["profile"], cfg["history_steps"])
    if cfg["calibration_confirmed"] is not True:
        raise ValueError("Fill measured frames/extrinsics/self_filter_boxes, then set calibration_confirmed=true")
    if cfg["pose_mode"] not in ("odometry", "imu"):
        raise ValueError("pose_mode must be odometry or imu")
    if set(cfg["frames"]) != {"world", "base", "imu", "output"} or not all(cfg["frames"].values()):
        raise ValueError("Declare every frame explicitly")
    if not cfg["lidar_extrinsics"]:
        raise ValueError("At least one calibrated LiDAR frame required")
    for frame, ext in cfg["lidar_extrinsics"].items():
        if not frame or set(ext) != {"translation_m", "quaternion_xyzw"}:
            raise ValueError("Invalid LiDAR extrinsic")
        finite_vector(ext["translation_m"], 3, "translation")
        rotation(ext["quaternion_xyzw"])
    rotation(cfg["imu_to_base_xyzw"])
    for box in cfg["self_filter_boxes"]:
        if np.asarray(box).shape != (2, 3) or not np.isfinite(box).all() or not np.all(np.asarray(box)[0] < np.asarray(box)[1]):
            raise ValueError("Self filter boxes must be [min_xyz,max_xyz] in body frame")
    EstimatorConfig(**cfg["estimator"]).validate()
    expected = {"publish_hz", "max_cloud_points", "max_cloud_bytes", "pose_buffer_size", "pose_max_gap_s",
                "input_timeout_s", "future_tolerance_s", "scan_duration_s", "point_time_field", "point_time_scale_s", "point_time_mode",
                "deskew_bin_s", "scan_stamp_reference", "pose_std", "max_pose_speed_m_s", "max_pose_rate_rad_s", "imu_hold_s",
                "undeskewed_speed_floor_m_s", "undeskewed_rate_floor_rad_s", "max_publish_query_age_s", "mapping_region", "fit_anchor_mode", "mapper_cloud_topic", "cloud_input_timeout_s"}
    r = cfg["runtime"]
    r.setdefault("point_time_mode", "relative")
    r.setdefault("mapping_region", "all")
    r.setdefault("fit_anchor_mode", "scan_header")
    r.setdefault("mapper_cloud_topic", cfg["topics"]["cloud"])
    r.setdefault("cloud_input_timeout_s", r["input_timeout_s"])
    # Older profiles retain their consumer limit. Deployment profiles reserve
    # an explicit part of that deadline for delivery and the next SDK tick.
    r.setdefault("max_publish_query_age_s", cfg["aos"]["max_alignment_age_s"])
    if set(r) != expected:
        raise ValueError("Runtime missing/unknown keys: " + str(set(r)^expected))
    if type(r["mapping_region"]) is not str or r["mapping_region"] not in ("all", "front"):
        raise ValueError("mapping_region must be all or front")
    if r["fit_anchor_mode"] not in ("scan_header", "latest_pose"):
        raise ValueError("fit_anchor_mode must be scan_header or latest_pose")
    if r["fit_anchor_mode"] == "latest_pose" and cfg["pose_mode"] != "odometry":
        raise ValueError("Latest-pose fitting requires measured odometry")
    for name, value in r.items():
        if name in ("point_time_field", "point_time_mode", "pose_std", "scan_stamp_reference", "mapping_region", "fit_anchor_mode", "mapper_cloud_topic"):
            continue
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid runtime option " + name)
    for name in ("max_cloud_points", "max_cloud_bytes", "pose_buffer_size"):
        if type(r[name]) is not int:
            raise ValueError(name + " must be int")
    if r["cloud_input_timeout_s"] > cfg["estimator"]["max_map_age_s"]:
        raise ValueError("Cloud input timeout must not exceed measurement lifetime")
    if r["scan_duration_s"]/r["deskew_bin_s"] > 200 or r["max_cloud_points"] > 1000000 or r["pose_buffer_size"] > 4096:
        raise ValueError("Runtime capacity exceeds bounded CPU limits")
    if r["scan_stamp_reference"] not in ("start", "end"):
        raise ValueError("scan_stamp_reference must be start or end")
    if r["point_time_mode"] not in ("relative", "absolute") or (r["point_time_mode"] == "absolute" and not r["point_time_field"]):
        raise ValueError("Absolute point time requires a declared field")
    std = finite_vector(r["pose_std"], 6, "pose_std")
    if np.any(std <= 0):
        raise ValueError("Unknown pose covariance requires positive calibrated floors")
    topics = cfg["topics"]
    if set(topics) != {"cloud", "odom", "imu", "terrain", "proprio", "policy", "status"} or not all(isinstance(v, str) and v.startswith("/") for v in topics.values()):
        raise ValueError("Declare absolute ROS topic names")
    if len(set(topics.values())) != len(topics):
        raise ValueError("Input/output topics must be distinct")
    mapper_topic = r["mapper_cloud_topic"]
    if not isinstance(mapper_topic, str) or re.fullmatch(r"(?:/[A-Za-z_][A-Za-z_0-9]*)+", mapper_topic) is None:
        raise ValueError("mapper_cloud_topic must be an absolute ROS topic name")
    if mapper_topic != topics["cloud"]:
        if mapper_topic in set(topics.values()) | {"/m20/gos/cloud_receipt", "/m20/gos/native_status", "/IMU", "/JOINTS_DATA", "/JOINTS_CMD"}:
            raise ValueError("Mapper cloud topic aliases another input/output")
        if r["mapping_region"] != "front" or "raw_input" not in cfg:
            raise ValueError("Separate mapper cloud requires the native front mapping pipeline")
    aos = cfg["aos"]
    aos.setdefault("coverage_hold_enabled", True)
    aos.setdefault("terrain_timing_hold_enabled", True)
    if set(aos) != {"model_contract", "model_path", "max_alignment_age_s", "max_proprio_age_s", "min_valid_fraction", "coverage_hold_enabled", "terrain_timing_hold_enabled"}:
        raise ValueError("AOS config missing/unknown keys")
    for name in ("coverage_hold_enabled", "terrain_timing_hold_enabled"):
        if type(aos[name]) is not bool:
            raise ValueError(name + " must be bool")
    for name in ("max_alignment_age_s", "max_proprio_age_s", "min_valid_fraction"):
        if not isinstance(aos[name], (int, float)) or not math.isfinite(aos[name]) or aos[name] <= 0:
            raise ValueError("Invalid AOS option: " + name)
    if aos["min_valid_fraction"] > 1:
        raise ValueError("min_valid_fraction must be <= 1")
    if aos["max_alignment_age_s"] > cfg["estimator"]["max_map_age_s"]:
        raise ValueError("Alignment limit must not exceed map age limit")
    if r["max_publish_query_age_s"] > aos["max_alignment_age_s"]:
        raise ValueError("Publication query age must not exceed alignment limit")
    # Deployment bundles can move between boards without keeping workstation paths.
    for name in ("model_contract", "model_path"):
        if aos[name]:
            model = Path(aos[name]).expanduser()
            aos[name] = str((model if model.is_absolute() else path.parent/model).resolve())
    return cfg


class Pipeline:
    def __init__(self, cfg):
        self.cfg, self.runtime = cfg, cfg["runtime"]
        self.mapper = ElevationMapper(cfg["profile"], EstimatorConfig(**cfg["estimator"]), retain=cfg["pose_mode"] == "odometry")
        self.poses = PoseBuffer(self.runtime["pose_buffer_size"], self.runtime["pose_max_gap_s"])
        self.last_scan = None
        self.last_pose_discontinuity = None
        self.output_mask = np.ones(len(GRID), dtype=bool)
        if self.runtime.get("mapping_region", "all") == "front":
            if "visibility" not in cfg:
                raise ValueError("Front map output requires calibrated sensor visibility")
            self.output_mask = front_map_mask(GRID, cfg["visibility"])

    def reset(self):
        self.mapper.reset()
        self.poses.clear()
        self.last_scan = None

    def add_pose(self, pose):
        if self.poses.poses:
            previous = self.poses.poses[-1]
            dt = pose.stamp-previous.stamp
            angle = 2*np.arccos(np.clip(abs(np.dot(pose.quaternion, previous.quaternion)), 0, 1))
            distance = float(np.linalg.norm(pose.position-previous.position))
            translation_limit = self.runtime["max_pose_speed_m_s"]*dt+.03
            rotation_limit = self.runtime["max_pose_rate_rad_s"]*dt+.03
            if dt <= 0 or distance > translation_limit or angle > rotation_limit:
                self.last_pose_discontinuity = {
                    "previous_stamp": previous.stamp, "stamp": pose.stamp, "dt_s": dt,
                    "translation_m": distance, "translation_limit_m": translation_limit,
                    "rotation_rad": float(angle), "rotation_limit_rad": rotation_limit,
                    "time_reversed": bool(dt <= 0),
                    "translation_exceeded": bool(distance > translation_limit),
                    "rotation_exceeded": bool(angle > rotation_limit),
                    "previous_position": previous.position.tolist(), "position": pose.position.tolist(),
                    "previous_quaternion_xyzw": previous.quaternion.tolist(),
                    "quaternion_xyzw": pose.quaternion.tolist()}
                for key in ("dt_s", "translation_m", "translation_limit_m", "rotation_rad", "rotation_limit_rad"):
                    value = self.last_pose_discontinuity[key]
                    if not np.isfinite(value):
                        self.last_pose_discontinuity[key] = None
                self.reset()
                raise ValueError("Pose time/jump discontinuity: map reset")
            pose.speed = max(pose.speed, float(np.linalg.norm(pose.position-previous.position)/dt))
            pose.angular_speed = max(pose.angular_speed, float(angle/dt))
        self.poses.add(pose)

    def ingest_message(self, msg, stamp, now, processing_now=None):
        started = time.monotonic()
        timings = {}
        r = self.runtime
        ext = self.cfg["lidar_extrinsics"].get(msg.header.frame_id)
        if ext is None:
            raise ValueError("Uncalibrated cloud frame: " + msg.header.frame_id)
        points, times = read_cloud(msg, r["max_cloud_points"], r["max_cloud_bytes"], r["point_time_field"], r["point_time_scale_s"], r.get("point_time_mode", "relative"))
        validate_cloud_time(stamp, now, r, times)
        stage_points = {"decoded": len(points)}
        good = np.isfinite(points).all(1)
        if times is not None:
            times = times[good]
        points = points[good]
        stage_points["finite"] = len(points)
        timings["decode"] = (time.monotonic()-started)*1000
        last_processing_time = now
        def fresh_mask(sample_times):
            nonlocal last_processing_time
            current = now if processing_now is None else float(processing_now())
            if not np.isfinite(current) or current < last_processing_time-r['future_tolerance_s']:
                raise ValueError('Processing clock nonfinite/reversed')
            last_processing_time = max(last_processing_time, current)
            return current-(stamp+sample_times) <= self.mapper.cfg.max_map_age_s

        # An already expired return cannot support a current cell. Filtering is
        # by absolute acquisition age only, never by height or a time watermark.
        if times is not None:
            fresh = fresh_mask(times)
            points,times = points[fresh],times[fresh]
        stage_points["capture_age"] = len(points)
        ranges = np.linalg.norm(points, axis=1)
        pose = self.poses.at(stamp)
        scan_pose = pose
        fit_pose = self.poses.poses[-1] if r.get("fit_anchor_mode", "scan_header") == "latest_pose" else pose
        # The final estimator can never use out-of-range points. Cull them
        # before the expensive per-time-bin transforms, preserving time indices.
        keep = (ranges >= self.mapper.cfg.min_range_m) & (ranges <= self.mapper.cfg.max_range_m)
        points,ranges = points[keep],ranges[keep]
        times = times[keep] if times is not None else None
        stage_points["range"] = len(points)
        body = points @ rotation(ext["quaternion_xyzw"]).T + np.asarray(ext["translation_m"])
        if r.get("mapping_region", "all") == "front":
            front = body[:, 0] > 0.
            points, ranges, body = points[front], ranges[front], body[front]
            times = times[front] if times is not None else None
        stage_points["front"] = len(points)
        keep = np.ones(len(points), dtype=bool)
        for lo, hi in self.cfg["self_filter_boxes"]:
            keep &= ~np.all((body >= lo) & (body <= hi), axis=1)
        stage_points["self_filter"] = int(keep.sum())
        displacement = 0.
        translation_from_scan, rotation_path = 0., 0.
        if times is not None and len(times) and self.cfg['pose_mode']=='odometry':
            lo,hi = stamp+min(0.,float(times.min())),stamp+max(0.,float(times.max()))
            # Translation is linearly interpolated. The norm on each segment
            # is bounded by its endpoints, even when measured speed exceeds a
            # configured bound. Include the actual interpolation boundaries.
            motion_poses = [self.poses.at(lo)]
            motion_poses += [p for p in self.poses.poses if lo < p.stamp < hi]
            if hi > lo:
                motion_poses.append(self.poses.at(hi))
            positions = np.asarray([p.position for p in motion_poses])
            # Bound distance to the actual fitting anchor, which may be newer
            # than the scan header. The original point times still drive deskew.
            displacement = float(np.max(np.linalg.norm(positions-fit_pose.position,axis=1)))
            translation_from_scan = float(np.max(np.linalg.norm(positions-scan_pose.position,axis=1)))
            # Total geodesic path length bounds every interpolated orientation's
            # distance to the header pose, including an arc that bends back.
            rotation_path = min(math.pi, sum(2*math.acos(float(np.clip(abs(np.dot(a.quaternion,b.quaternion)),0.,1.)))
                                              for a,b in zip(motion_poses,motion_poses[1:])))
        elif self.cfg['pose_mode']=='imu':
            displacement = r['max_pose_speed_m_s']*r['scan_duration_s']
        else:
            displacement = float(np.linalg.norm(scan_pose.position-fit_pose.position))
        radius = math.sqrt(.85**2+.55**2+max(abs(self.mapper.cfg.min_relative_z_m),abs(self.mapper.cfg.max_relative_z_m))**2)
        # Rotation cannot change ||body||. Reverse triangle inequality proves
        # every point outside this sphere is outside the final ROI at anchor.
        keep &= np.linalg.norm(body,axis=1) <= radius+displacement+1e-9
        if self.cfg['pose_mode'] == 'odometry':
            # A conservative box in the REAL fit anchor can reject points
            # before the expensive per-bin transforms. Bound movement with the
            # measured interpolation path, not the commanded walking speed.
            basis = yaw_rotation(fit_pose.yaw)
            approximate = body @ (scan_pose.R.T @ basis) + (scan_pose.position-fit_pose.position) @ basis
            error = translation_from_scan + 2*np.linalg.norm(body,axis=1)*math.sin(rotation_path*.5) + 1e-7
            keep &= ((np.abs(approximate[:,0]) <= .85+error) &
                     (np.abs(approximate[:,1]) <= .55+error) &
                     (approximate[:,2] >= self.mapper.cfg.min_relative_z_m-error) &
                     (approximate[:,2] <= self.mapper.cfg.max_relative_z_m+error))
        points, ranges = points[keep], ranges[keep]
        times = times[keep] if times is not None else None
        stage_points["motion_roi"] = len(points)
        if 'visibility' in self.cfg:
            in_view=visible_body(body[keep],self.cfg['visibility'])
            points,ranges=points[in_view],ranges[in_view]
            times=times[in_view] if times is not None else None
        stage_points["visibility"] = len(points)
        timings["prefilter"] = (time.monotonic()-started)*1000-timings["decode"]
        deskew_started = time.monotonic()
        if times is not None and self.cfg["pose_mode"] == "odometry":
            world, pose = deskew(points, times, pose, self.poses, ext["translation_m"], ext["quaternion_xyzw"],
                                 r["deskew_bin_s"], return_quality=True)
            span = r["deskew_bin_s"]
        else:
            world = to_world(points, pose, ext["translation_m"], ext["quaternion_xyzw"])
            span = r["scan_duration_s"]
        timings["deskew"] = (time.monotonic()-deskew_started)*1000
        if fit_pose is not scan_pose:
            # Crop/fit around the newest REAL pose supplied to this job, rather
            # than shrinking forward lookahead around the old scan origin.
            # Never replace acquisition times with this newer anchor time.
            variance = np.maximum(pose.variance, fit_pose.variance)
            variance[3:] = np.max(variance[3:])
            pose = Pose(fit_pose.stamp, fit_pose.position, fit_pose.quaternion, variance,
                        max(pose.speed, fit_pose.speed), max(pose.angular_speed, fit_pose.angular_speed))
        # Bound unresolved temporal motion. IMU-only has no measured translation.
        speed = max(pose.speed, r["undeskewed_speed_floor_m_s"])
        rate = max(pose.angular_speed, r["undeskewed_rate_floor_rad_s"])
        if self.cfg["pose_mode"] == "imu":
            speed = r["max_pose_speed_m_s"]
        radius = self.mapper.cfg.max_range_m
        motion_std = (speed+radius*rate)*span/math.sqrt(12)
        # The geometric anchor time and the oldest supported measurement time differ.
        # Keep the original global time for diagnostics; cell support retains
        # its own oldest real acquisition instead of borrowing unrelated times.
        capture_stamp = stamp + (float(np.min(times)) if times is not None and len(times) else
                                 (-r["scan_duration_s"] if r["scan_stamp_reference"] == "end" else 0.))
        # Read the real processing clock again after deskew, then recompute all
        # fits and uncertainty from the surviving support. Final cell aging at
        # publication still rejects samples that expire during the fit itself.
        if times is not None:
            fresh = fresh_mask(times)
            world,ranges,times = world[fresh],ranges[fresh],times[fresh]
        stage_points["post_deskew_age"] = len(world)
        fit_started = time.monotonic()
        self.mapper.ingest(world, ranges, pose, motion_std, capture_stamp,
                           sample_stamps=stamp+times if times is not None else None,
                           scan_stamp=stamp)
        timings["fit"] = (time.monotonic()-fit_started)*1000
        if 'visibility' in self.cfg:
            patch=self.mapper.patches[-1]
            xy=GRID @ yaw_rotation(pose.yaw)[:2,:2].T
            body_centres=np.column_stack((xy,patch.z-pose.position[2])) @ pose.R
            visible=visible_body(body_centres,self.cfg['visibility'])
            patch.valid &= visible
            self.mapper.stats['valid_cells']=int(patch.valid.sum())
        self.mapper.stats["pipeline_timings_ms"] = timings
        self.mapper.stats["point_filter_stages"] = stage_points
        self.mapper.stats["deskew_points"] = len(points)
        self.mapper.stats["fit_anchor_mode"] = r.get("fit_anchor_mode", "scan_header")
        self.mapper.stats["fit_anchor_stamp"] = pose.stamp
        self.mapper.stats["scan_header_stamp"] = stamp
        self.last_scan = scan_pose

    def snapshot(self, now):
        if not self.poses.poses:
            raise ValueError("No pose")
        pose = self.poses.poses[-1]
        if now-pose.stamp > self.runtime["input_timeout_s"] or pose.stamp-now > self.runtime["future_tolerance_s"]:
            raise ValueError("Pose stale/future")
        if self.last_scan is None:
            raise ValueError("No accepted scan")
        if self.cfg["pose_mode"] == "imu":
            pose = self.last_scan
            if now-pose.stamp > self.runtime["imu_hold_s"]:
                raise ValueError("IMU-only scan expired; translation cannot be reprojected")
        channels, var, age = self.mapper.snapshot(pose)
        if self.runtime.get("mapping_region", "all") == "front":
            # Match front_only after every delayed/history reprojection.
            outside = ~self.output_mask
            channels[:, outside] = 0.
            var[outside], age[outside] = 0., -1.
        return pose, (channels, var, age)
