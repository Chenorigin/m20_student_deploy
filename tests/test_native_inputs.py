"""Pure native input ABI, bounded fragmented Unix framing and latched identity."""
import copy
import json
from pathlib import Path
import struct
import pytest
from m20_heightmap.native_packet import (SOURCE,PROTOCOL,NativeIdentity,receive_packet,
                                         validate_native_meta,validate_receipt)
from m20_heightmap.runtime import load_config


def sample():
    blob=struct.pack('<6f',1,2,3,4,5,6)
    return {'source':SOURCE,'sec':10,'nanosec':0,'width':2,'height':1,'point_step':12,'row_step':24,
        'bytes':24,'driver_pid':1824,'writer_guid':'010ff773200704a80100000000000103',
        'frame':'lidar_link','bigendian':False,'dense':True,'native_session':'ab'*16,'native_sequence':1,
        'fields':[{'name':n,'offset':i*4,'datatype':7,'count':1} for i,n in enumerate('xyz')]},blob


class FragmentedInput:
    def __init__(self,data):self.data=data;self.bytes_read=0
    def recv_into(self,view):
        n=min(len(view),3,len(self.data));view[:n]=self.data[:n];self.data=self.data[n:];self.bytes_read+=n;return n


def test_native_packet_fragmentation_and_bit_exact_payload():
    meta,blob=sample();text=json.dumps(meta).encode()
    assert receive_packet(FragmentedInput(struct.pack('<I',len(text))+text+blob))==(meta,blob)
    sock=FragmentedInput(struct.pack('<I',4097)+b'x'*5000)
    with pytest.raises(ValueError):receive_packet(sock)
    assert sock.bytes_read==4
    wrong=dict(meta,bytes=16000001);text=json.dumps(wrong).encode();sock=FragmentedInput(struct.pack('<I',len(text))+text+b'ignored')
    with pytest.raises(ValueError):receive_packet(sock)
    assert sock.data==b'ignored'


@pytest.mark.parametrize('key,bad',[('source','AOS_local_native_UDP'),('native_session','bad'),
    ('native_sequence',True),('native_sequence',0),('native_sequence',2**64),('driver_pid',1825),
    ('writer_guid','010ff773200704a80000000000000103')])
def test_factory_source_identity_rejects_and_latches(key,bad):
    meta,blob=sample();wrong=dict(meta);wrong[key]=bad;gate=NativeIdentity()
    with pytest.raises(ValueError):gate.accept(wrong,len(blob))
    with pytest.raises(ValueError):gate.accept(meta,len(blob))


def test_source_restart_and_replay_never_reset_identity():
    meta,blob=sample()
    for update in ({},{'native_sequence':0},{'native_sequence':2,'native_session':'cd'*16},
                   {'native_sequence':2,'writer_guid':'010ff773200705a80100000000000103'}):
        gate=NativeIdentity();gate.accept(meta,len(blob))
        with pytest.raises(ValueError):gate.accept(dict(meta,**update),len(blob))
        with pytest.raises(ValueError):gate.accept(dict(meta,native_sequence=3),len(blob))
    gate=NativeIdentity();gate.accept(meta,len(blob));gate.accept(dict(meta,native_sequence=7),len(blob))
    assert gate.sequence==7


def test_receipt_checks_real_native_source_and_freshness():
    cfg=json.loads((Path(__file__).parents[1]/'deploy/heightmap_v1.json').read_text())
    meta,_=sample();validate_native_meta(meta,24)
    r={'protocol':PROTOCOL,'source':SOURCE,'board':'CA9B_GOS','payload_validated':True,'published_local':True,
       'session':'cd'*16,'source_session':meta['native_session'],'driver_pid':1824,'writer_guid':meta['writer_guid'],
       'sequence':1,'stamp':10.,'end_stamp':10.1,'received_at':10.12,'frame':'lidar_link','points':2,'raw_bytes':24,
       'mapper_topic':cfg['runtime']['mapper_cloud_topic'],'mapper_subset_published':True,
       'mapper_points':2,'mapper_raw_bytes':24,'mapper_reference_points':0}
    assert validate_receipt(r,cfg,10.15)==(10.,10.1)
    for k,v in [('protocol','tcp_lz4_v1'),('payload_validated',False),('driver_pid',1),
                ('source_session','bad'),('sequence',0),('end_stamp',9.),('received_at',11.),
                ('stamp',float('nan')),('writer_guid','010ff773200704a80000000000000103')]:
        bad=copy.deepcopy(r);bad[k]=v
        with pytest.raises(ValueError):validate_receipt(bad,cfg,10.15)
    bad=copy.deepcopy(r);del bad['mapper_points']
    with pytest.raises(ValueError,match='mapper subset'):validate_receipt(bad,cfg,10.15)


def test_config_rejects_wrong_native_source_contract(tmp_path):
    cfg=json.loads((Path(__file__).parents[1]/'deploy/heightmap_v1.json').read_text())
    cfg['raw_input']['source']='AOS_local_native_UDP';p=tmp_path/'config.json';p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='native DDS'):load_config(p)
