"""Content-bound geometry/physical-check metadata, without ROS or hardware calls."""
import hashlib
import json
from pathlib import Path

def geometry_hash(root, config_path):
    root,config_path=Path(root),Path(config_path)
    digest=hashlib.sha256()
    for path in (config_path,root/'deploy/robot.json',root/'deploy/site.json',root/'deploy/gos_lio.yaml'):
        data=path.read_bytes();digest.update(len(data).to_bytes(8,'little'));digest.update(data)
    return digest.hexdigest()

def commission_matches(root, config_path, profile):
    root,config_path=Path(root),Path(config_path)
    try:
        receipt=json.loads((root/'runtime/commissioned.json').read_text())
        hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('config_sha256',config_path),('robot_sha256',root/'deploy/robot.json'),('site_sha256',root/'deploy/site.json'),('lio_sha256',root/'deploy/gos_lio.yaml')]}
        return (receipt.get('profile')==profile and receipt.get('boot_id')==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and
                all(receipt.get(key)==value for key,value in hashes.items()) and
                all(receipt.get(key) is True for key in ('factory_geometry_checked','software_limits_checked','operator_controls_start','physical_estop_available')))
    except (OSError,ValueError):return False

def loaded_geometry_commissioned(root, config_path, profile, loaded_hash):
    """Edits and re-commissioning cannot bless a process holding old parameters."""
    try:
        return (bool(loaded_hash) and geometry_hash(root,config_path)==loaded_hash and
                commission_matches(root,config_path,profile))
    except OSError:return False
