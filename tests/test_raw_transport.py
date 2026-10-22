"""Lossless transport and fail-closed receiver receipts, with real local sockets."""
import copy
import json
from pathlib import Path
import socket
import struct
import threading
import pytest
from m20_heightmap.raw_transport import (Codec, HEADER, MAGIC, MAX_BODY, MAX_WIRE,
    LatestServer, inspect_header, receive_frame, validate_receipt, endpoint_from_status)


def packet():
    return {'source':'AOS_local_native_UDP','sec':12,'nanosec':34,'width':2,'height':1,
        'point_step':12,'row_step':24,'bytes':24,'driver_pid':3515,
        'writer_guid':'010f1aebbb0d91100000000000000103','frame':'lidar_link',
        'bigendian':False,'dense':True,'fields':[{'name':n,'offset':i*4,'datatype':7,'count':1} for i,n in enumerate('xyz')],
        'session':'12'*16,'sequence':1}, struct.pack('<6f',1,2,3,4,5,6)


def test_bit_exact_raw_scan_and_fragmented_socket():
    codec=Codec();meta,blob=packet();frame=codec.encode(meta,blob)
    assert codec.decode(frame[:HEADER.size],frame[HEADER.size:])==(meta,blob)
    a,b=socket.socketpair()
    def writer():
        with a:
            for i in range(0,len(frame),3):a.sendall(frame[i:i+3])
    t=threading.Thread(target=writer);t.start()
    with b:assert receive_frame(b,codec)[0]==(meta,blob)
    t.join(1);assert not t.is_alive()


@pytest.mark.parametrize('magic,wire,raw',[(b'nope',1,5),(MAGIC,MAX_WIRE+1,5),(MAGIC,1,MAX_BODY+1),(MAGIC,1,0)])
def test_lengths_rejected_before_read_or_allocation(magic,wire,raw):
    with pytest.raises(ValueError):inspect_header(HEADER.pack(magic,wire,raw,0))


def test_crc_corrupt_data_and_truncation_rejected():
    codec=Codec();frame=bytearray(codec.encode(*packet()))
    frame[12]^=1
    with pytest.raises(ValueError,match='CRC'):codec.decode(bytes(frame[:HEADER.size]),bytes(frame[HEADER.size:]))
    frame=codec.encode(*packet())
    with pytest.raises(ValueError):codec.decode(frame[:HEADER.size],frame[HEADER.size:-1])
    a,b=socket.socketpair()
    a.sendall(frame[:-1]);a.close()
    with b,pytest.raises(ValueError,match='closed'):receive_frame(b,codec)


def test_schema_source_and_native_writer_pid_fail_closed():
    codec=Codec();meta,blob=packet()
    for key,bad in [('source','other'),('driver_pid',3516),('bytes',23),('width',True),('height',0),('dense',1)]:
        wrong=copy.deepcopy(meta);wrong[key]=bad
        with pytest.raises(ValueError):codec.encode(wrong,blob)


def test_latest_server_never_replays_retained_packet_on_connect():
    codec=Codec();server=LatestServer('127.0.0.1');meta,blob=packet()
    try:
        server.submit(codec.encode(meta,blob))
        with socket.create_connection(('127.0.0.1',server.port),timeout=1) as client:
            client.settimeout(.1)
            with pytest.raises(socket.timeout):client.recv(1)
            meta['sequence']=2;server.submit(codec.encode(meta,blob))
            client.settimeout(1)
            assert receive_frame(client,codec)[0]==(meta,blob)
    finally:server.close()
    assert not server.thread.is_alive()


def test_endpoint_and_gos_per_frame_receipt_validation():
    cfg=json.loads((Path(__file__).parents[1]/'deploy/heightmap_v1.json').read_text())
    for bad in (None,[],1,'not an object'):
        with pytest.raises(ValueError):endpoint_from_status(bad)
        with pytest.raises(ValueError):validate_receipt(bad,cfg,10.2)
    endpoint={'board':'CA9B_AOS','source':'selected_local_rslidar_udp','transport':'tcp_lz4_v1',
        'address':'192.0.2.1','port':43210,'session':'ab'*16}
    assert endpoint_from_status(endpoint)==('192.0.2.1',43210,'ab'*16)
    for key,value in [('address','0.0.0.0'),('address','127.0.0.1'),('port',True),('session','bad')]:
        wrong=dict(endpoint);wrong[key]=value
        with pytest.raises(ValueError):endpoint_from_status(wrong)
    receipt={'protocol':'tcp_lz4_v1','board':'CA9B_GOS','crc_verified':True,'published_local':True,
        'stamp':10.,'end_stamp':10.18,'received_at':10.19,'frame':'lidar_link','sequence':2,
        'points':2,'raw_bytes':24,'session':'ab'*16}
    assert validate_receipt(receipt,cfg,10.2)==(10.,10.18)
    for key,value in [('end_stamp',9.),('stamp',11.),('received_at',11.),('crc_verified',False),('published_local',False),('points',0),('sequence',0),('end_stamp',float('nan'))]:
        wrong=dict(receipt);wrong[key]=value
        with pytest.raises(ValueError):validate_receipt(wrong,cfg,10.2)
