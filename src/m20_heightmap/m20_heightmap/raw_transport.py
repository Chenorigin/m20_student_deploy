"""Bounded lossless raw-scan framing; no ROS, motor, or policy dependencies.

LZ4 block API: https://github.com/lz4/lz4/blob/dev/lib/lz4.h
CRC detects corruption; it is not authentication. Use only the robot's trusted link.
"""
import ctypes
import ctypes.util
import ipaddress
import json
import math
import socket
import struct
import threading
import zlib
import numpy as np

MAX_META = 4096
MAX_BLOB = 16000000
MAX_BODY = 4 + MAX_META + MAX_BLOB
MAX_WIRE = MAX_BODY + MAX_BODY // 255 + 32
HEADER = struct.Struct('<4sIII')
MAGIC = b'M2S1'


class Codec:
    def __init__(self):
        library = ctypes.util.find_library('lz4')
        if not library:
            raise RuntimeError('System liblz4 is required for raw scan transport')
        self.lib = ctypes.CDLL(library)
        self.lib.LZ4_compressBound.argtypes = [ctypes.c_int]
        self.lib.LZ4_compressBound.restype = ctypes.c_int
        for name in ('LZ4_compress_default', 'LZ4_decompress_safe'):
            fn = getattr(self.lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            fn.restype = ctypes.c_int

    def encode(self, meta, blob):
        validate_meta(meta, len(blob))
        text = json.dumps(meta, separators=(',', ':'), allow_nan=False).encode()
        if not 1 <= len(text) <= MAX_META:
            raise ValueError('raw metadata budget exceeded')
        body = struct.pack('<I', len(text)) + text + blob
        # Reversible byte planes expose repeated high bytes in XYZ/time fields.
        # This changes only the compression representation, never any point.
        step = meta['point_step']
        count = len(blob)//step
        end = count*step
        shuffled = np.frombuffer(blob[:end],dtype=np.uint8).reshape(count,step).T.copy().tobytes()+blob[end:]
        wire_body = body[:4+len(text)]+shuffled
        target = ctypes.create_string_buffer(self.lib.LZ4_compressBound(len(body)))
        size = self.lib.LZ4_compress_default(wire_body, target, len(wire_body), len(target))
        if not 1 <= size <= MAX_WIRE:
            raise ValueError('raw compression failed/budget exceeded')
        return HEADER.pack(MAGIC, size, len(body), zlib.crc32(body)) + target.raw[:size]

    def decode(self, header, data):
        size, raw_size, crc = inspect_header(header)
        if len(data) != size:
            raise ValueError('raw frame truncated')
        output = ctypes.create_string_buffer(raw_size)
        count = self.lib.LZ4_decompress_safe(data, output, size, raw_size)
        if count != raw_size:
            raise ValueError('invalid LZ4 frame')
        body = output.raw
        n = struct.unpack('<I', body[:4])[0]
        if not 1 <= n <= MAX_META or n + 4 > len(body):
            raise ValueError('invalid raw metadata length')
        meta = json.loads(body[4:4+n])
        shuffled = body[4+n:]
        validate_meta(meta, len(shuffled))
        step = meta['point_step']
        count = len(shuffled)//step
        end = count*step
        blob = np.frombuffer(shuffled[:end],dtype=np.uint8).reshape(step,count).T.copy().tobytes()+shuffled[end:]
        if zlib.crc32(body[:4+n]+blob) != crc:
            raise ValueError('raw frame CRC mismatch')
        return meta, blob


def inspect_header(header):
    if len(header) != HEADER.size:
        raise ValueError('raw header truncated')
    magic, size, raw_size, crc = HEADER.unpack(header)
    if magic != MAGIC or not 1 <= size <= MAX_WIRE or not 5 <= raw_size <= MAX_BODY:
        raise ValueError('raw protocol/size budget rejected')
    return size, raw_size, crc


def validate_meta(meta, amount, expected_source='AOS_local_native_UDP'):
    if expected_source not in ('AOS_local_native_UDP','GOS_factory_native_UDP'):
        raise ValueError('unsupported native source contract')
    if not isinstance(meta, dict) or meta.get('source') != expected_source:
        raise ValueError('raw source rejected')
    for key in ('sec', 'nanosec', 'width', 'height', 'point_step', 'row_step', 'bytes', 'driver_pid'):
        if type(meta.get(key)) is not int:
            raise ValueError('raw integer required: ' + key)
    if (not 0 <= meta['nanosec'] < 1000000000 or not 0 < amount <= MAX_BLOB or
            meta['bytes'] != amount or not 0 < meta['width']*meta['height'] <= 200000 or
            meta['width'] <= 0 or meta['height'] <= 0 or meta['point_step'] <= 0 or
            meta['row_step'] < meta['width']*meta['point_step'] or
            meta['row_step']*meta['height'] != amount or not 0 < meta['driver_pid'] < 65536):
        raise ValueError('raw dimensions/time/payload rejected')
    for key in ('dense', 'bigendian'):
        if type(meta.get(key)) is not bool:
            raise ValueError('raw boolean required')
    if not isinstance(meta.get('frame'), str) or not 1 <= len(meta['frame']) <= 256:
        raise ValueError('raw frame rejected')
    guid = meta.get('writer_guid')
    if not isinstance(guid, str) or len(guid) != 32:
        raise ValueError('raw writer GUID rejected')
    try:
        b = bytes.fromhex(guid)
    except ValueError as exc:
        raise ValueError('raw writer GUID rejected') from exc
    if len(b) != 16 or int.from_bytes(b[4:6], 'little') != meta['driver_pid']:
        raise ValueError('raw writer/PID mismatch')
    fields = meta.get('fields')
    if not isinstance(fields, list) or not 1 <= len(fields) <= 16:
        raise ValueError('raw fields rejected')
    names = set()
    sizes = {1:1, 2:1, 3:2, 4:2, 5:4, 6:4, 7:4, 8:8}
    for field in fields:
        if (not isinstance(field, dict) or not isinstance(field.get('name'), str) or
                not field['name'] or field['name'] in names or
                any(type(field.get(k)) is not int for k in ('offset','count','datatype')) or
                field['datatype'] not in sizes or field['count'] <= 0 or field['offset'] < 0 or
                field['offset'] + sizes[field['datatype']]*field['count'] > meta['point_step']):
            raise ValueError('raw point field rejected')
        names.add(field['name'])


def exact(sock, count):
    data = bytearray(count)
    view = memoryview(data)
    offset = 0
    while offset < count:
        n = sock.recv_into(view[offset:])
        if not n:
            raise ValueError('raw stream closed during frame')
        offset += n
    return bytes(data)


def receive_frame(sock, codec):
    header = exact(sock, HEADER.size)
    size, _, _ = inspect_header(header)  # Reject lengths before allocation/read.
    return codec.decode(header, exact(sock, size)), size + HEADER.size


def endpoint_from_status(value):
    if not isinstance(value,dict):
        raise ValueError('raw endpoint object required')
    if (value.get('board') != 'CA9B_AOS' or value.get('source') != 'selected_local_rslidar_udp' or
            value.get('transport') != 'tcp_lz4_v1'):
        raise ValueError('raw endpoint source rejected')
    address = ipaddress.ip_address(value['address'])
    if address.version != 4 or address.is_unspecified or address.is_multicast or address.is_loopback:
        raise ValueError('raw endpoint address rejected')
    port, session = value.get('port'), value.get('session')
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError('raw endpoint port rejected')
    if not isinstance(session, str) or len(session) != 32 or len(bytes.fromhex(session)) != 16:
        raise ValueError('raw endpoint session rejected')
    return str(address), port, session


def validate_receipt(value, cfg, now):
    """AOS probes the real GOS decode event without pulling a large DDS scan back."""
    r = cfg['runtime']
    if not isinstance(value,dict):
        raise ValueError('GOS cloud receipt object required')
    if (value.get('protocol') != 'tcp_lz4_v1' or value.get('board') != 'CA9B_GOS' or
            value.get('crc_verified') is not True or value.get('published_local') is not True):
        raise ValueError('GOS cloud receipt rejected')
    for name in ('stamp','end_stamp','received_at'):
        if type(value.get(name)) not in (int,float) or not math.isfinite(value[name]):
            raise ValueError('GOS cloud receipt nonfinite time')
    stamp, end, received = value['stamp'], value['end_stamp'], value['received_at']
    if (now-end > r['input_timeout_s'] or now-received > r['input_timeout_s'] or
            max(stamp,end,received)-now > r['future_tolerance_s'] or
            abs(end-stamp) > r['scan_duration_s']):
        raise ValueError('GOS cloud receipt stale/future')
    if (value.get('frame') not in cfg['lidar_extrinsics'] or
            type(value.get('sequence')) is not int or value['sequence'] <= 0 or
            type(value.get('points')) is not int or not 0 < value['points'] <= r['max_cloud_points'] or
            type(value.get('raw_bytes')) is not int or not 0 < value['raw_bytes'] <= r['max_cloud_bytes']):
        raise ValueError('GOS cloud receipt dimensions/sequence rejected')
    session = value.get('session')
    if not isinstance(session,str) or len(session) != 32 or len(bytes.fromhex(session)) != 16:
        raise ValueError('GOS cloud receipt session rejected')
    return float(stamp), float(end)


class LatestServer:
    """One client, one pending frame. Slow clients disconnect, never queue scans."""
    def __init__(self, address):
        if ipaddress.ip_address(address).is_unspecified:
            raise ValueError('bind only to the resolved board link')
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((address, 0))
        self.socket.listen(1)
        self.socket.settimeout(.2)
        self.address, self.port = self.socket.getsockname()
        self.condition = threading.Condition()
        self.latest = None
        self.sequence = 0
        self.stopping = False
        self.client = None
        self.error = None
        self.sent = 0
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self.condition:
            self.latest = frame
            self.sequence += 1
            self.condition.notify_all()

    def serve(self):
        while not self.stopping:
            try:
                client, _ = self.socket.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self.client = client
            try:
                client.settimeout(.15)
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # Do not let a reconnect replay an old retained scan.
                with self.condition:
                    last = self.sequence
                while not self.stopping:
                    with self.condition:
                        self.condition.wait_for(lambda: self.stopping or self.sequence != last, timeout=.2)
                        if self.stopping:
                            break
                        if self.sequence == last:
                            continue
                        frame, last = self.latest, self.sequence
                    client.sendall(frame)
                    self.sent += 1
            except OSError as exc:
                self.error = str(exc)
            finally:
                client.close()
                self.client = None

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.socket.close()
        client = self.client
        if client is not None:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.thread.join(timeout=2)
