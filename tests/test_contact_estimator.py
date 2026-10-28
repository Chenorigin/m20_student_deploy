"""Independent rigid-body oracle and fail-closed passive capture regressions."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/m20_heightmap'))
from m20_heightmap.contact_estimator import ContactModel, rpy_rotation

spec = importlib.util.spec_from_file_location('capture_contact', ROOT/'scripts/capture_cts_contact.py')
capture = importlib.util.module_from_spec(spec); spec.loader.exec_module(capture)
spec2 = importlib.util.spec_from_file_location('compare_contact', ROOT/'scripts/compare_cts_contact.py')
comparison = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(comparison)


def geometry():
    return json.loads((ROOT/'deploy/m20_contact_geometry.json').read_text())


def oracle(document):
    root = ET.Element('mujoco')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', gravity='0 0 -9.81')
    world = ET.SubElement(root, 'worldbody')
    fmt = lambda x: ' '.join(map(str, np.asarray(x).reshape(-1)))
    for leg in document['legs']:
        parent = world
        for joint in leg['joints']:
            # M20 URDF joint origins have zero rotations; assert this rather
            # than silently letting the independent oracle omit them.
            assert joint['origin']['rpy'] == [0, 0, 0]
            body = ET.SubElement(parent, 'body', name=joint['child'], pos=fmt(joint['origin']['xyz']))
            ET.SubElement(body, 'joint', name=joint['name'], type='hinge', axis=fmt(joint['axis']))
            ri = rpy_rotation(joint['com']['rpy']); inertia = ri@np.asarray(joint['inertia'])@ri.T
            # Supply principal axes explicitly: MuJoCo's fullinertia compiler
            # diagonalization approximates these very small off-diagonal terms.
            eigenvalues, axes = np.linalg.eigh(inertia)
            if np.linalg.det(axes) < 0: axes[:,0] *= -1
            quat = np.empty(4); mujoco.mju_mat2Quat(quat,axes.reshape(-1))
            ET.SubElement(body, 'inertial', mass=str(joint['mass']), pos=fmt(joint['com']['xyz']),
                          diaginertia=fmt(eigenvalues),quat=fmt(quat))
            parent = body
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def state():
    q = np.tile([.1, -.5, 1.1, .2], 4)
    return dict(q=q, dq=np.zeros(16), ddq=np.zeros(16), base_rotation=np.eye(3),
                base_specific_force=[0,0,9.81], omega_body=[0,0,0], alpha_body=[0,0,0])


def test_inverse_dynamics_and_material_point_jacobian_against_mujoco():
    document = geometry(); core = ContactModel(document)
    model = oracle(document); data = mujoco.MjData(model); rng = np.random.default_rng(12703)
    for _ in range(40):
        q = rng.uniform(-1.2,1.2,16); dq = rng.normal(0,1,16); ddq = rng.normal(0,2,16)
        specific = rng.normal(0,1,3)+[0,0,9.81]
        model.opt.gravity[:] = -specific
        data.qpos[:] = q; data.qvel[:] = dq
        mujoco.mj_forward(model,data); data.qacc[:] = ddq; mujoco.mj_inverse(model,data)
        actual = core.dynamics(q,dq,ddq,np.eye(3),specific,[0,0,0],[0,0,0])
        np.testing.assert_allclose(actual['model_torque_sdk_nm'], data.qfrc_inverse, atol=2e-10, rtol=2e-10)
        for leg, item in enumerate(document['legs']):
            body = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,item['joints'][-1]['child'])
            point = actual['support_relative_world_m'][leg]
            jac = np.zeros((3,model.nv)); mujoco.mj_jac(model,data,jac,None,point,body)
            np.testing.assert_allclose(actual['point_jacobian_world_m_per_rad'][leg], jac[:,leg*4:leg*4+3], atol=1e-12)


def test_known_external_force_sign_and_unvalidated_validity():
    core = ContactModel(geometry()); s=state()
    base = core.dynamics(**s); torque = base['model_torque_sdk_nm'].copy()
    force = np.array([4.,-3.,30.])
    for leg,jac in enumerate(base['point_jacobian_world_m_per_rad']):
        torque[4*leg:4*leg+3] -= jac.T@force
    result=core.diagnose(torque,**s)
    np.testing.assert_allclose(result['estimated_force_world_N'],[force]*4,atol=1e-10)
    assert result['contact_valid']==[False]*4 and result['hardware_validated'] is False
    # A tempting sign error is detected: positive motor residual is downward.
    negative=core.diagnose(2*base['model_torque_sdk_nm']-torque,**s)
    np.testing.assert_allclose(negative['estimated_force_world_N'],[-force]*4,atol=1e-10)


def test_world_rotation_covariance():
    core=ContactModel(geometry()); s=state(); a=core.dynamics(**s)
    r=rpy_rotation([.3,-.2,.4]); s['base_rotation']=r; b=core.dynamics(**s)
    np.testing.assert_allclose(a['model_torque_sdk_nm'],b['model_torque_sdk_nm'],atol=1e-12)
    # Cylinder support moves with world vertical; verify torque covariance only.


def test_singular_and_nonfinite_rejected():
    core=ContactModel(geometry()); s=state(); s['q']=np.zeros(16)
    result=core.diagnose(np.zeros(16),**s)
    assert result['contact_valid']==[False]*4
    assert any(x=='ill_conditioned_support_jacobian' for x in result['invalid_reason'])
    json.dumps(result,allow_nan=False)
    s['q'][0]=np.nan
    with pytest.raises(ValueError,match='q'):core.diagnose(np.zeros(16),**s)


@pytest.mark.parametrize('mutation', ['mass','inertia','order'])
def test_invalid_model_rejected(mutation):
    g=geometry()
    if mutation=='mass':g['legs'][0]['joints'][0]['mass']=-1
    if mutation=='inertia':g['legs'][0]['joints'][0]['inertia'][0][0]=-1
    if mutation=='order':g['joint_order'].reverse()
    with pytest.raises(ValueError):ContactModel(g)


def test_capture_only_receives_and_retains_source_identity():
    # The delivered entry itself contains no command publisher construction.
    import ast
    tree=ast.parse((ROOT/'scripts/capture_cts_contact.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert any(n.func.attr=='create_subscription' for n in calls)
    assert not any(n.func.attr in ('create_publisher','publish') for n in calls)
    assert '/JOINTS_CMD' not in capture.TOPICS


def test_bad_source_time_and_quaternion():
    with pytest.raises(ValueError):capture.header_stamp({'header':{'stamp':{'sec':0,'nanosec':-1}}})
    with pytest.raises(ValueError):capture.quaternion_rotation([0,0,0,0])
    with pytest.raises(ValueError):capture.quaternion_rotation([0,0,0,float('nan')])


def test_nonfinite_raw_feedback_preserved_without_json_failure():
    import struct
    bad = {'torque':[float('nan'),float('inf'),-float('inf'),0.]}
    encoded = capture.json_safe(bad)
    json.dumps(encoded,allow_nan=False)
    values = [struct.unpack('!d',bytes.fromhex(item['nonfinite_float64_be_hex']))[0]
              for item in encoded['torque'][:3]]
    assert np.isnan(values[0]) and values[1]==float('inf') and values[2]==-float('inf')
    assert encoded['torque'][3]==0.


def test_motor_order_and_timestamp_regression_rejected():
    robot=json.loads((ROOT/'deploy/robot.json').read_text())
    diag=capture.Diagnostics(robot,geometry())
    payload={'header':{'stamp':{'sec':1,'nanosec':0}},'data':{'joints_data':[
        dict(name=list(str(i).encode())+[0],position=0.,velocity=0.,torque=0.) for i in range(16)]}}
    assert diag.consume('/JOINTS_DATA',payload,10**9) is None
    wrong=copy.deepcopy(payload);wrong['data']['joints_data'][0]['name']=[49,0]
    with pytest.raises(ValueError,match='identity'):diag.consume('/JOINTS_DATA',wrong,10**9)
    imu={'header':{'stamp':{'sec':1,'nanosec':0}},'orientation':dict(x=0.,y=0.,z=0.,w=1.),
         'angular_velocity':dict(x=0.,y=0.,z=0.),'linear_acceleration':dict(x=0.,y=0.,z=9.81)}
    diag.consume('/IMU',imu,10**9)
    with pytest.raises(ValueError,match='clock'):diag.consume('/IMU',imu,10**9+1000)


def pair_fixture(path, condition, force):
    # Test-only claim documents exercise comparison validation; no authority is
    # granted even for perfectly separated synthetic force distributions.
    import hashlib
    path.mkdir()
    rows=[dict(receive_monotonic_ns=10**9+i*100_000_000,
               diagnostic=dict(estimated_force_world_N=[[0,0,force]]*4)) for i in range(40)]
    raw=''.join(json.dumps(row)+'\n' for row in rows).encode()
    (path/'samples.jsonl').write_bytes(raw)
    manifest=dict(origin='hardware_recording',condition=condition,labelled_leg='FL',
                  samples_sha256=hashlib.sha256(raw).hexdigest(),publisher_set_stable=True,limits_reached=False,
                  hardware_serial='TEST_ONLY',boot_id='TEST_ONLY',robot_sha256='a'*64,
                  geometry_sha256='b'*64,publishers_before={t:[{'node':'TEST_ONLY','gid':[1]*24}]
                      for t in ('/JOINTS_DATA','/IMU')})
    (path/'manifest.json').write_text(json.dumps(manifest))
    return path


def test_labelled_pair_rejects_overlap_and_never_grants_admission(tmp_path):
    air=pair_fixture(tmp_path/'air','wheel_unloaded',1.)
    loaded=pair_fixture(tmp_path/'loaded','wheel_loaded',30.)
    result=comparison.compare(air,loaded)
    assert result['diagnostic_distributions_separated'] is True
    assert result['hardware_validated'] is False and result['hardware_admission_granted'] is False
    overlap=pair_fixture(tmp_path/'overlap','wheel_loaded',1.)
    assert comparison.compare(air,overlap)['diagnostic_distributions_separated'] is False


@pytest.mark.parametrize('mutation',['digest','leg','boot','lying'])
def test_pair_tamper_identity_and_labels_rejected(tmp_path,mutation):
    air=pair_fixture(tmp_path/'air','wheel_unloaded',1.)
    loaded=pair_fixture(tmp_path/'loaded','wheel_loaded',30.)
    path=loaded/'manifest.json';m=json.loads(path.read_text())
    m[{'digest':'samples_sha256','leg':'labelled_leg','boot':'boot_id','lying':'condition'}[mutation]] = {
        'digest':'WRONG','leg':'FR','boot':'OTHER','lying':'lying'}[mutation]
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):comparison.compare(air,loaded)


@pytest.mark.parametrize('key',['hardware_serial','boot_id','robot_sha256','geometry_sha256','publishers_before'])
def test_pair_rejects_identity_missing_on_both_sides(tmp_path,key):
    air=pair_fixture(tmp_path/'air','wheel_unloaded',1.)
    loaded=pair_fixture(tmp_path/'loaded','wheel_loaded',30.)
    for folder in (air,loaded):
        path=folder/'manifest.json';m=json.loads(path.read_text());del m[key]
        path.write_text(json.dumps(m))
    with pytest.raises(ValueError):comparison.compare(air,loaded)
