from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
def test_recovery_guard_and_actual_handset_parser(tmp_path):
    sdk=ROOT/'src/m20_student_sdk'
    includes=['plugin','vendor/include/types','vendor/include/utils','vendor/interface/user_command','vendor/interface/network']
    for name in ['policy_fault_recovery_test','gamepad_recovery_test']:
        exe=tmp_path/name
        subprocess.run(['g++','-std=c++17','-O0','-pthread','-I/usr/include/eigen3']+['-I'+str(sdk/p) for p in includes]+[str(ROOT/'tests'/f'{name}.cpp'),'-o',str(exe)],check=True,capture_output=True,text=True,timeout=60)
        r=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10)
        assert r.returncode==0,r.stdout+r.stderr

def test_guard_oracle_rejects_unconditional_reset(tmp_path):
    sdk=ROOT/'src/m20_student_sdk'
    text=(sdk/'vendor/include/utils/policy_fault_recovery.hpp').read_text()
    start='const bool accepted=active_&&ack_&&ReadyLocked(now)&&now>=ack_time_&&now-ack_time_<=std::chrono::seconds(5);'
    assert start in text
    (tmp_path/'policy_fault_recovery.hpp').write_text(text.replace(start,'const bool accepted=true;'))
    exe=tmp_path/'broken_guard'
    subprocess.run(['g++','-std=c++17','-pthread','-I'+str(tmp_path),str(ROOT/'tests/policy_fault_recovery_test.cpp'),'-o',str(exe)],check=True,capture_output=True,text=True,timeout=30)
    r=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10)
    assert r.returncode!=0,'guard verifier accepted unconditional reset'
