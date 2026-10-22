#!/usr/bin/env python3
"""Derive robot parameters from pinned training YAML and the user's reference SDK.

No imports of the SDK, no hardware initialization, no ROS or motor commands.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src/m20_heightmap"))
from m20_heightmap.contract import JOINT_NAMES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpp_array(text, pattern, names=False):
    match = re.search(pattern, text, re.S)
    if match is None:
        raise ValueError("Reference declaration not found: " + pattern)
    body = match.group(1)
    values = re.findall(r'"([^"\n]+)"', body) if names else [float(x.strip()) for x in body.split(",") if x.strip()]
    if len(values) != 16:
        raise ValueError("Reference joint arrays must have exactly 16 elements")
    return values


def regex_values(mapping, names):
    result = []
    for name in names:
        found = [float(value) for pattern, value in mapping.items() if re.fullmatch(pattern, name)]
        if len(found) != 1:
            raise ValueError("Expected one numeric rule for " + name)
        result.append(found[0])
    return result


def derive(env_path, reference):
    deploy = reference/"M20_perception_deploy_63d"
    runner = deploy/"src/M20_sdk_deploy/run_policy/m20_policy_runner.hpp"
    interface = deploy/"src/M20_sdk_deploy/interface/robot/hardware/m20_interface.hpp"
    dds = deploy/"src/M20_sdk_deploy/interface/robot/hardware/dds_interface.hpp"
    # BaseLoader reads scalar data without constructing Python objects in tagged training YAML.
    env = yaml.load(env_path.read_text(), Loader=yaml.BaseLoader)
    rtext, itext = runner.read_text(), interface.read_text()
    sdk_order = cpp_array(rtext, r'robot_order\s*=\s*\{(.*?)\};', True)
    policy_order = cpp_array(rtext, r'policy_order\s*=\s*\{(.*?)\};', True)
    if tuple(policy_order) != JOINT_NAMES or set(sdk_order) != set(JOINT_NAMES):
        raise ValueError("Reference policy joint order differs from student ABI")
    direction = cpp_array(itext, r'float\s+joint_dir\[16\]\s*=\s*\{(.*?)\};')
    offset = cpp_array(itext, r'float\s+init_pos_offset\[16\]\s*=\s*\{(.*?)\};')
    default = regex_values(env["scene"]["robot"]["init_state"]["joint_pos"], policy_order)
    leg_scale = regex_values(env["actions"]["joint_pos"]["scale"], policy_order[:12])
    scale = leg_scale+[float(env["actions"]["joint_vel"]["scale"])]*4
    actuator = env["scene"]["robot"]["actuators"]
    kp = [float(actuator["wheel" if "wheel" in n else "joint"]["stiffness"]) for n in sdk_order]
    kd = [float(actuator["wheel" if "wheel" in n else "joint"]["damping"]) for n in sdk_order]
    # Check PD and action scales against the pinned SDK, not just training defaults.
    for field, values in (("kp_", kp[:4]), ("kd_", kd[:4])):
        pattern = re.escape(field)+r'\s*=\s*Vec4f\(([^)]*)\)'
        found = re.search(pattern, rtext)
        if found is None or [float(v) for v in found.group(1).split(",")] != values:
            raise ValueError("Training/reference PD mismatch: " + field)
    reference_scales = cpp_array(rtext, r'action_scale_robot\s*=\s*\{(.*?)\};')
    if reference_scales != [scale[policy_order.index(n)] for n in sdk_order]:
        raise ValueError("Training/reference action scale mismatch")
    processed_clip = [list(map(float, env["actions"]["joint_pos" if i < 12 else "joint_vel"]["clip"][".*"])) for i in range(16)]
    reference_default = cpp_array(rtext, r'dof_default_eigen_policy\s*<<\s*(.*?);')
    policy_dt = float(env["sim"]["dt"])*int(env["decimation"])
    observations = env["observations"]["policy"]
    files = [env_path, runner, interface, dds, deploy/"config/m20_superlio_pgtt.yaml"]
    rev = subprocess.run(["git", "-C", str(reference), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return {
        "schema_version": 1,
        "sources": {"reference_url": "https://arcusgit.vip.cpolar.cn/Dawning/m20_pro_perceptive_lidar",
                    "reference_local_checkout": str(reference), "reference_commit": rev,
                    "remote_latest_verified": False, "teacher_env_yaml": str(env_path),
                    "sha256": {str(p): sha(p) for p in files}},
        "policy": {"dt_s": policy_dt, "nominal_joint_servo_dt_s_from_training": float(env["sim"]["dt"]),
                   "decimation_from_training": int(env["decimation"]),
                   "normalized_action_clip": None,
                   "processed_target_clip_policy": processed_clip,
                   "previous_action": "previous issued normalized 16D action; not mapped joint target",
                   "runtime_noise_injection": False},
        "observation": {"scales": {name: float(observations[name]["scale"]) for name in
                                    ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")},
                        "clip": [-100., 100.], "wheel_positions_zero": True},
        "joints": {"policy_order": policy_order, "sdk_order": sdk_order,
                   "policy_from_sdk_indices": [sdk_order.index(n) for n in policy_order],
                   "sdk_from_policy_indices": [policy_order.index(n) for n in sdk_order],
                   "default_position_policy_rad": default, "action_scale_policy": scale,
                   "kp_sdk": kp, "kd_sdk": kd, "sdk_direction": direction,
                   "sdk_initial_offset_deg": offset,
                   "offset_wrap_rule": "reference M20Interface.ResetPositionOffset may adjust hipy/knee offsets by +/-360 degrees; use effective SDK values",
                   "state_formula": "q_model_sdk=q_raw*direction+offset_rad; dq_model_sdk=dq_raw*direction",
                   "command_formula": "q_raw=(q_target_model_sdk-offset_rad)*direction; dq_raw=dq_target_model_sdk*direction",
                   "sdk_api_already_applies_offset_direction": True,
                   "reference_legacy_default_position_policy_rad": reference_default,
                   "reference_default_is_used": False},
        "hardware": {"joint_state_topic": "/JOINTS_DATA", "joint_state_type": "drdds/msg/JointsData",
                     "joint_command_topic": "/JOINTS_CMD", "joint_command_type": "drdds/msg/JointsDataCmd",
                     "direct_imu_topic": "/IMU", "direct_imu_type": "sensor_msgs/msg/Imu",
                     "bridged_imu_topic": "/M20_IMU_BRIDGE", "bridged_imu_type": "drdds/msg/ImuData",
                     "bridged_rpy_unit": "degrees; convert once to radians", "direct_quaternion_order": "xyzw",
                     "control_word_motor_control": 4,
                     "sdk_command_matrix_columns": ["kp", "position", "kd", "velocity", "torque_ff"],
                     "sdk_lifecycle_can_publish_commands": True,
                     "limits_verified_on_hardware": False,
                     "joint_position_limits_sdk_rad": None,
                     "joint_velocity_limits_sdk_rad_s": None},
        "training_limits": {"leg_effort_nm": float(actuator["joint"]["effort_limit"]),
                            "wheel_effort_nm": float(actuator["wheel"]["effort_limit"]),
                            "leg_velocity_rad_s": float(actuator["joint"]["velocity_limit"]),
                            "wheel_velocity_rad_s": float(actuator["wheel"]["velocity_limit"])},
        "command_training_ranges": {key: list(map(float, env["commands"]["base_velocity"]["ranges"][key])) for key in
                                    ("lin_vel_x", "lin_vel_y", "ang_vel_z")},
        "pending": ["actual AOS/GOS topic/frame/DDS and synchronized timestamps",
                    "LiDAR-to-base and IMU-to-base measured extrinsics; self filter geometry",
                    "world-to-trained-base odometry provider and child-frame adaptation",
                    "per-point timestamp field/unit/header start-end semantics",
                    "real sensor and relative pose covariance calibration",
                    "SDK effective offsets, hardware hard limits and control takeover/exit procedure",
                    "GOS ARM processing budget under actual co-resident load"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--teacher-env", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=ROOT/"deploy/robot.json")
    args = ap.parse_args()
    cfg = derive(args.teacher_env.resolve(), args.reference.resolve())
    for profile in ("v0", "v1"):
        exported = json.loads((ROOT/f"models/{profile}/deployment.json").read_text())
        provenance = exported["training_provenance"]
        teacher = Path(provenance["teacher_checkpoint"])
        if teacher.parent/"params/env.yaml" != args.teacher_env.resolve() or sha(teacher) != provenance["teacher_sha256"]:
            raise ValueError("Teacher source/hash is not the checkpoint's pinned teacher")
        clip = exported["training_provenance"]["training_config"]["clip_actions"]
        if cfg["policy"]["normalized_action_clip"] not in (None, [-clip, clip]):
            raise ValueError("Students have different action clipping; require per-model config")
        cfg["policy"]["normalized_action_clip"] = [-clip, clip]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cfg, indent=2)+"\n")
    for profile in ("v0", "v1"):
        template = json.loads((ROOT/f"src/m20_heightmap/config/{profile}.json").read_text())
        template['raw_input']={'protocol':'native_dds_factory_gos_v1','source':'GOS_factory_native_UDP'}
        template['visibility']={'site_config':'../deploy/site.json'}
        template['estimator']['max_map_age_s']=.5
        template['topics'].update(cloud='/m20/gos/points',imu='/m20/gos/imu')
        template["aos"]["model_contract"] = f"../models/{profile}/observation_contract.json"
        template["aos"]["model_path"] = f"../models/{profile}/policy.pt"
        (args.output.parent/f"heightmap_{profile}.json").write_text(json.dumps(template, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "policy_dt_s": cfg["policy"]["dt_s"],
                      "reference_commit": cfg["sources"]["reference_commit"],
                      "legacy_pose_overridden_with_training_pose": cfg["joints"]["default_position_policy_rad"] != cfg["joints"]["reference_legacy_default_position_policy_rad"]}, indent=2))


if __name__ == "__main__":
    main()
