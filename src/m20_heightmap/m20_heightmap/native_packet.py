"""GOS factory DDS source contract and its bounded local Unix IPC framing.

The shared exact()/metadata helpers do not create sockets or TCP endpoints.
No compression is applied to this native DDS input path.
"""
import math
import struct
from .raw_transport import MAX_META, validate_meta, exact

SOURCE = 'GOS_factory_native_UDP'
PROTOCOL = 'native_dds_factory_gos_v1'


def hex_identity(value):
    if not isinstance(value,str) or len(value)!=32:
        raise ValueError('native source session identity rejected')
    try:
        if len(bytes.fromhex(value))!=16:raise ValueError('native source session identity rejected')
    except ValueError as exc:
        raise ValueError('native source session identity rejected') from exc
    return value


def validate_native_meta(meta, amount):
    validate_meta(meta,amount,SOURCE)
    if bytes.fromhex(meta['writer_guid'])[8:12]!=b'\x01\x00\x00\x00':
        raise ValueError('GOS factory DDS participant contract changed')
    hex_identity(meta.get('native_session'))
    n=meta.get('native_sequence')
    if type(n) is not int or not 0<n<2**64:
        raise ValueError('native source sequence rejected')


def receive_packet(sock):
    import json
    n=struct.unpack('<I',exact(sock,4))[0]
    if not 1<=n<=MAX_META:raise ValueError('native metadata length rejected')
    meta=json.loads(exact(sock,n))
    if not isinstance(meta,dict) or type(meta.get('bytes')) is not int:
        raise ValueError('native payload length rejected')
    validate_native_meta(meta,meta['bytes'])  # Check every bound before allocation.
    return meta,exact(sock,meta['bytes'])


class NativeIdentity:
    """A malformed identity, restart, ambiguity or replay permanently latches."""
    def __init__(self):
        self.identity=None
        self.sequence=0
        self.fault=None

    def accept(self,meta,amount):
        if self.fault:raise ValueError(self.fault)
        try:
            validate_native_meta(meta,amount)
            identity=(meta['native_session'],meta['driver_pid'],meta['writer_guid'])
            if self.identity is not None and identity!=self.identity:
                raise ValueError('native factory source changed; restart after inspection')
            if meta['native_sequence']<=self.sequence:
                raise ValueError('native factory sequence repeated/regressed')
            self.identity,self.sequence=identity,meta['native_sequence']
        except (ValueError,TypeError,KeyError) as exc:
            self.fault=str(exc)
            raise ValueError(self.fault) from exc


def validate_receipt(value,cfg,now):
    if (not isinstance(value,dict) or value.get('protocol')!=PROTOCOL or
            value.get('source')!=SOURCE or value.get('board')!='CA9B_GOS' or
            value.get('payload_validated') is not True or value.get('published_local') is not True):
        raise ValueError('native DDS receipt source/validation rejected')
    for name in ('stamp','end_stamp','received_at'):
        if type(value.get(name)) not in (int,float) or not math.isfinite(value[name]):
            raise ValueError('native DDS receipt nonfinite time')
    r=cfg['runtime'];stamp,end,received=(value[n] for n in ('stamp','end_stamp','received_at'))
    if (now-end>r.get('cloud_input_timeout_s',r['input_timeout_s']) or now-received>r['input_timeout_s'] or
            max(stamp,end,received)-now>r['future_tolerance_s'] or
            abs(end-stamp)>r['scan_duration_s']):
        raise ValueError('native DDS receipt stale/future')
    if (value.get('frame') not in cfg['lidar_extrinsics'] or
            type(value.get('sequence')) is not int or not 0<value['sequence']<2**64 or
            type(value.get('points')) is not int or not 0<value['points']<=r['max_cloud_points'] or
            type(value.get('raw_bytes')) is not int or not 0<value['raw_bytes']<=r['max_cloud_bytes']):
        raise ValueError('native DDS receipt dimensions/sequence rejected')
    mapper_topic = r.get('mapper_cloud_topic', cfg['topics']['cloud'])
    if mapper_topic != cfg['topics']['cloud']:
        if (value.get('mapper_topic') != mapper_topic or value.get('mapper_subset_published') is not True or
                type(value.get('mapper_points')) is not int or not 0 <= value['mapper_points'] <= value['points'] or
                type(value.get('mapper_raw_bytes')) is not int or not 0 <= value['mapper_raw_bytes'] <= value['raw_bytes'] or
                type(value.get('mapper_reference_points')) is not int or not 0 <= value['mapper_reference_points'] <= min(1,value['mapper_points']) or
                (value['mapper_points'] == 0) != (value['mapper_raw_bytes'] == 0)):
            raise ValueError('native DDS receipt mapper subset rejected')
    for name in ('session','source_session','writer_guid'):hex_identity(value.get(name))
    pid=value.get('driver_pid')
    if (type(pid) is not int or not 0<pid<65536 or
            int.from_bytes(bytes.fromhex(value['writer_guid'])[4:6],'little')!=pid or
            bytes.fromhex(value['writer_guid'])[8:12]!=b'\x01\x00\x00\x00'):
        raise ValueError('native DDS receipt factory PID rejected')
    return float(stamp),float(end)
