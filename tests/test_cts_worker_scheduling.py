"""Validate production worker scheduling without hardware or RT privilege."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def worker_probe(tmp_path_factory):
    target = tmp_path_factory.mktemp('cts_worker') / 'probe'
    subprocess.run(['g++', '-std=c++17', '-O0', '-pthread',
                    '-I' + str(ROOT / 'src/m20_student_sdk/plugin'),
                    str(ROOT / 'tests/cts_worker_probe.cpp'), '-o', str(target)],
                   capture_output=True, text=True, check=True, timeout=30)
    return target


def invoke(probe, cpu=None, priority=None, mode='validate'):
    env = dict(os.environ)
    for name in ['M20_CTS_WORKER_CPU', 'M20_CTS_WORKER_FIFO_PRIORITY']:
        env.pop(name, None)
    if cpu is not None:
        env['M20_CTS_WORKER_CPU'] = str(cpu)
    if priority is not None:
        env['M20_CTS_WORKER_FIFO_PRIORITY'] = str(priority)
    return subprocess.run([str(probe), mode], env=env, capture_output=True,
                          text=True, timeout=5)


def test_disabled_does_not_change_scheduling(worker_probe):
    result = invoke(worker_probe, mode='apply')
    assert result.returncode == 0 and result.stdout.strip() == 'unconfigured'


def test_complete_configuration(worker_probe):
    result = invoke(worker_probe, min(os.sched_getaffinity(0)),
                    os.sched_get_priority_min(os.SCHED_FIFO))
    assert result.returncode == 0 and result.stdout.strip() == 'configured'


@pytest.mark.parametrize(('cpu', 'priority'), [
    (0, None), (None, 20), ('', 20), ('-1', 20), ('4x', 20), (1024, 20),
    ('999999999999999999999999', 20), (0, ''), (0, 0), (0, -1), (0, 100), (0, '20x'),
])
def test_invalid_configuration_rejected(worker_probe, cpu, priority):
    assert invoke(worker_probe, cpu, priority).returncode != 0


def test_cpu_outside_launch_affinity_rejected(worker_probe):
    allowed = os.sched_getaffinity(0)
    absent = next((cpu for cpu in range(1024) if cpu not in allowed), None)
    if absent is None:
        pytest.skip('launch allows every representable CPU')
    result = invoke(worker_probe, absent, 20, mode='apply')
    assert result.returncode != 0 and 'outside launch affinity' in result.stderr
