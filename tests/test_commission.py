import hashlib
import json
from pathlib import Path
from m20_heightmap.commission import geometry_hash, loaded_geometry_commissioned

def test_loaded_geometry_cannot_be_recommissioned_after_disk_edit(tmp_path):
    (tmp_path/'runtime').mkdir();(tmp_path/'deploy').mkdir()
    cfg=tmp_path/'runtime/heightmap_v1.local.json';cfg.write_text('{"candidate":1}')
    for name in ('robot.json','site.json','gos_lio.yaml'):
        (tmp_path/'deploy'/name).write_text(name)
    def receipt():
        data={'profile':'v1','boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        for key,path in [('config_sha256',cfg),('robot_sha256',tmp_path/'deploy/robot.json'),('site_sha256',tmp_path/'deploy/site.json'),('lio_sha256',tmp_path/'deploy/gos_lio.yaml')]:
            data[key]=hashlib.sha256(path.read_bytes()).hexdigest()
        for key in ('factory_geometry_checked','software_limits_checked','operator_controls_start','physical_estop_available'):data[key]=True
        (tmp_path/'runtime/commissioned.json').write_text(json.dumps(data))
    original=geometry_hash(tmp_path,cfg)
    assert not loaded_geometry_commissioned(tmp_path,cfg,'v1',original)
    receipt()
    assert loaded_geometry_commissioned(tmp_path,cfg,'v1',original)
    assert not loaded_geometry_commissioned(tmp_path,cfg,'v0',original)
    cfg.write_text('{"candidate":2}');receipt()
    assert not loaded_geometry_commissioned(tmp_path,cfg,'v1',original)
    assert loaded_geometry_commissioned(tmp_path,cfg,'v1',geometry_hash(tmp_path,cfg))
