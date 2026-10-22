import json
from pathlib import Path
import numpy as np
import pytest
from m20_heightmap.robot_config import (load_robot_config, raw_state_to_policy,
                                       corrected_state_to_policy, action_to_sdk_matrix,
                                       sdk_matrix_to_raw)
from m20_heightmap.runtime import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_raw_state_round_trip_and_mapping():
    cfg = load_robot_config(ROOT/"deploy/robot.json")
    names = cfg["joints"]["policy_order"]
    # Independent name-based reference, not inversion of the same permutation.
    q = np.arange(16, dtype=float)*.02
    dq = np.arange(16, dtype=float)*.03
    sdk = cfg["joints"]["sdk_order"]
    direction = cfg["joints"]["sdk_direction"]
    offsets = cfg["joints"]["sdk_initial_offset_deg"]
    raw_q = [(q[names.index(n)]-np.deg2rad(offsets[i]))*direction[i] for i,n in enumerate(sdk)]
    raw_dq = [dq[names.index(n)]*direction[i] for i,n in enumerate(sdk)]
    got_q, got_dq = raw_state_to_policy(raw_q, raw_dq, cfg)
    np.testing.assert_allclose(got_q, q, atol=1e-12)
    np.testing.assert_allclose(got_dq, dq, atol=1e-12)
    corrected_q = [q[names.index(n)] for n in sdk]
    corrected_dq = [dq[names.index(n)] for n in sdk]
    np.testing.assert_allclose(corrected_state_to_policy(corrected_q, corrected_dq, cfg)[0], q)
    # Hip/knee SDK initialization can choose a different 2*pi branch.
    wrapped = np.asarray(offsets)+360
    q2, _ = raw_state_to_policy(raw_q, raw_dq, cfg, wrapped)
    np.testing.assert_allclose(q2, q+2*np.pi, atol=1e-12)


def test_zero_action_training_pose_and_wheel_modes():
    cfg = load_robot_config(ROOT/"deploy/robot.json")
    matrix = action_to_sdk_matrix(np.zeros(16), cfg)
    # This known training pose differs from the SDK legacy pose (hipy -0.6/knee 1.0).
    np.testing.assert_allclose(matrix[0:4, 1], [0., -.3, .6, 0.])
    np.testing.assert_allclose(matrix[8:12, 1], [0., .3, -.6, 0.])
    assert not matrix[[3,7,11,15],0].any()
    assert not matrix[:,3:].any()


def test_action_scaling_and_raw_command_signs():
    cfg = load_robot_config(ROOT/"deploy/robot.json")
    action = np.zeros(16);action[0] = 2.;action[1] = .5;action[12] = 1.;action[13] = 2.
    matrix = action_to_sdk_matrix(action, cfg)
    np.testing.assert_allclose(matrix[0,1],.25)
    np.testing.assert_allclose(matrix[1,1],-.175)
    np.testing.assert_allclose(matrix[3,3],5.)
    np.testing.assert_allclose(matrix[7,3],10.)
    raw = sdk_matrix_to_raw(matrix, cfg)
    np.testing.assert_allclose(raw[7,3],-10.)
    np.testing.assert_allclose(raw[2,1],-(.6-np.deg2rad(160)),atol=1e-7)
    assert np.isfinite(raw).all()
    # The selected MuJoCo mapping has no post-scale wheel target clip.
    saturated = action_to_sdk_matrix(np.full(16, 1000.), cfg)
    assert saturated[3,3] == 500.
    # Legacy configs still apply their original processed-target clipping.
    cfg["policy"].pop("wheel_target_velocity_clip_enabled")
    assert action_to_sdk_matrix(np.full(16, 1000.), cfg)[3,3] == 100.
    with pytest.raises(ValueError): action_to_sdk_matrix(np.full(16,np.nan),cfg)


def test_bad_config_rejected(tmp_path):
    cfg=json.loads((ROOT/"deploy/robot.json").read_text())
    cfg["joints"]["sdk_direction"][0]=0
    path=tmp_path/"bad.json";path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):load_robot_config(path)


def test_portable_model_paths(tmp_path):
    cfg=json.loads((ROOT/"deploy/heightmap_v1.json").read_text())
    cfg["calibration_confirmed"]=True
    path=tmp_path/"deployment"/"config.json";path.parent.mkdir();path.write_text(json.dumps(cfg))
    site=tmp_path/'deploy/site.json';site.parent.mkdir();site.write_bytes((ROOT/'deploy/site.json').read_bytes())
    got=load_config(path)
    assert got["aos"]["model_path"]==str((tmp_path/"models/v1/policy.onnx").resolve())
    assert got["aos"]["model_contract"]==str((tmp_path/"models/v1/observation_contract.json").resolve())
    assert got['visibility']['horizontal_fov_deg']==135.
