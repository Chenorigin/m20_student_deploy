import os
import importlib.util
import json
from pathlib import Path
import shutil
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('alignment',ROOT/'scripts/check_cts_runtime_alignment.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@pytest.fixture
def release(tmp_path):
    # A real exported residual contract and measured site configuration are inputs.
    value=os.environ.get('CTS_ALIGNMENT_FIXTURE')
    if not value:pytest.skip('Set CTS_ALIGNMENT_FIXTURE to a prepared real release')
    source=Path(value)
    for name in ['deploy','runtime','models']:
        shutil.copytree(source/name,tmp_path/name)
    return tmp_path


def test_real_configuration_and_capture(release):
    result=module.check_config(release)
    assert result['capture']['capture_points']==667
    assert result['output_points']==187


@pytest.mark.parametrize('change',['wide_command','hold_missing','session_hold','capture_missing','range','mapping','pd','model_path'])
def test_legacy_configuration_rejected(release,change):
    filename='deploy/robot.json' if change in ['wide_command','hold_missing','session_hold','pd'] else 'runtime/heightmap_v1.local.json'
    path=release/filename;data=json.loads(path.read_text())
    if change=='wide_command':data['command_operator_ranges']['lin_vel_x']=[-1.2,1.2]
    if change=='hold_missing':del data['policy']['hold_enabled']
    if change=='session_hold':data['policy']['terrain_source_session_hold_enabled']=True
    if change=='capture_missing':del data['estimator']['map_capture_padding_m']
    if change=='range':data['estimator']['max_range_m']=3.
    if change=='mapping':data['runtime']['mapping_region']='all'
    if change=='model_path':data['aos']['model_path']='/different/policy.onnx'
    if change=='pd':data['joints']['kp_sdk'][0]=100.
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):module.check_config(release)


def test_build_manifest_rejects_missing_or_mismatched_installed_consumer(tmp_path):
    source=tmp_path/'src/m20_heightmap/m20_heightmap'
    installed=tmp_path/'install/m20_heightmap/lib/python3.8/site-packages/m20_heightmap'
    source.mkdir(parents=True);installed.mkdir(parents=True)
    (source/'runtime.py').write_text('expected')
    (installed/'runtime.py').write_text('old')
    with pytest.raises(ValueError,match='Installed Python differs'):module.build_files(tmp_path,'GOS')
    (installed/'runtime.py').write_text('expected')
    with pytest.raises(ValueError,match='native lidar'):module.build_files(tmp_path,'GOS')
    native=tmp_path/'install/m20_native_inputs/lib/m20_native_inputs/native_lidar_input'
    native.parent.mkdir(parents=True);native.write_bytes(b'fixture executable')
    before=module.build_files(tmp_path,'GOS')
    native.write_bytes(b'changed executable')
    assert module.build_files(tmp_path,'GOS')!=before


def test_launcher_checks_boards_before_deploy(release,monkeypatch):
    import sys
    import types
    sys.path.insert(0,str(ROOT/'scripts'))
    import start_cts_stair_noise as launch
    training=release/'training'
    entry=training/'scripts/tools/m20_robot.py';entry.parent.mkdir(parents=True);entry.touch()
    calls=[]
    class Board:
        root='/isolated/release'
        def ssh(self,command):
            calls.append(command)
            raise RuntimeError('stale build identity')
    def loader(module):
        module.parser=lambda: types.SimpleNamespace(parse_args=lambda argv:None)
        module.deploy=lambda *args:calls.append('UNSAFE_DEPLOY')
        module.main=lambda argv:module.deploy(None,Board(),Board(),None)
    monkeypatch.setattr(launch.importlib.util,'spec_from_file_location',lambda *args:types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=loader)))
    monkeypatch.setattr(launch.importlib.util,'module_from_spec',lambda spec:types.SimpleNamespace())
    with pytest.raises(RuntimeError,match='stale build'):
        launch.main(['--training',str(training),'--bundle',str(release/'models/v1'),'--aos','aos','--gos','gos'])
    assert len(calls)==1 and 'check_cts_runtime_alignment.py' in calls[0]
    assert 'UNSAFE_DEPLOY' not in calls
