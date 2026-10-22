"""Native GOS factory DDS -> private local IPC -> local ROS2 inputs/receipts.

No AOS raw relay, network point-cloud client, model or motor interface.
"""
import hashlib
import json
from pathlib import Path
import socket
import struct
import threading
import time
import uuid
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import PointCloud2,Imu
from std_msgs.msg import String
from .native_packet import SOURCE,PROTOCOL,NativeIdentity,receive_packet
from .raw_packet import cloud_message,subset_cloud
from .cloud import mapper_input_mask
from .ros_nodes import sensor_qos
from .runtime import load_config


class NativeInputs(Node):
    def __init__(self):
        super().__init__('m20_gos_native_inputs')
        if Path('/opt/robot/os_type').read_text().strip()!='CA9B_GOS':
            raise ValueError('GOS factory native inputs only')
        config_path=Path(self.declare_parameter('config','').value).resolve()
        self.cfg=load_config(config_path)
        if self.cfg.get('raw_input',{}).get('protocol')!=PROTOCOL:
            raise ValueError('native DDS source configuration required')
        self.path=self.declare_parameter('socket_path','').value
        self.binary=(config_path.parent.parent/'install/m20_native_inputs/lib/m20_native_inputs/native_lidar_input').resolve()
        self.binary_sha=hashlib.sha256(self.binary.read_bytes()).digest()
        self.peer_identity=None
        q=sensor_qos(1);q.reliability=ReliabilityPolicy.RELIABLE
        self.clouds=self.create_publisher(PointCloud2,self.cfg['topics']['cloud'],q)
        self.mapper_topic=self.cfg['runtime'].get('mapper_cloud_topic',self.cfg['topics']['cloud'])
        self.mapper_clouds=(self.create_publisher(PointCloud2,self.mapper_topic,q)
                            if self.mapper_topic!=self.cfg['topics']['cloud'] else None)
        self.last_cloud_metrics={'mapper_topic':self.mapper_topic,'mapper_points':None,
            'mapper_raw_bytes':None,'mapper_subset_published':False,'crop_ms':None,
            'mapper_reference_points':0,'raw_points':None,'raw_bytes':None}
        iq=sensor_qos(4);iq.reliability=ReliabilityPolicy.RELIABLE
        self.imus=self.create_publisher(Imu,self.cfg['topics']['imu'],iq)
        self.receipts=self.create_publisher(String,'/m20/gos/cloud_receipt',1)
        self.status=self.create_publisher(String,'/m20/gos/native_status',1)
        self.imu_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Imu,'/IMU',self.imu,sensor_qos(8),
                                 callback_group=self.imu_group)
        self.identity=NativeIdentity();self.session=uuid.uuid4().hex
        self.lock=threading.Lock();self.latest=None;self.error=None
        self.count=0;self.imu_count=0;self.stop=threading.Event();self.source_fault=threading.Event();self.sock=None
        self.worker=threading.Thread(target=self.reader,daemon=True);self.worker.start()
        self.create_timer(.005,self.publish_latest);self.create_timer(1.,self.report)

    def verify_peer(self,sock):
        pid,uid,_=struct.unpack('3i',sock.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,struct.calcsize('3i')))
        if pid<=0 or uid!=0:raise ValueError('native IPC peer must be the verified root input executable')
        proc=Path('/proc')/str(pid)
        if (proc/'exe').resolve()!=self.binary or hashlib.sha256((proc/'exe').read_bytes()).digest()!=self.binary_sha:
            raise ValueError('native IPC executable identity changed')
        ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
        identity=(pid,ticks)
        if self.peer_identity is not None and self.peer_identity!=identity:
            raise ValueError('native IPC process restarted; restart perception after inspection')
        self.peer_identity=identity

    def reader(self):
        while not self.stop.is_set() and not self.source_fault.is_set():
            try:
                with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
                    self.sock=sock;sock.settimeout(1.);sock.connect(self.path);self.verify_peer(sock)
                    while not self.stop.is_set():
                        meta,blob=receive_packet(sock);self.identity.accept(meta,len(blob))
                        with self.lock:self.latest=(meta,blob)
            except (ValueError,KeyError,TypeError) as exc:
                self.error=str(exc);self.source_fault.set()
                with self.lock:self.latest=None
            except OSError as exc:
                self.error=str(exc);self.stop.wait(.1)
            finally:self.sock=None

    def publish_latest(self):
        if self.source_fault.is_set():return
        with self.lock:packet,self.latest=self.latest,None
        if packet is None:return
        meta,blob=packet
        try:
            now=self.get_clock().now().nanoseconds*1e-9
            decoded=cloud_message(meta,blob,self.cfg,now,expected_source=SOURCE,
                                  return_points=self.mapper_clouds is not None)
            msg,end=decoded[:2]
            if self.get_clock().now().nanoseconds*1e-9-end>self.cfg['runtime']['cloud_input_timeout_s']:
                raise ValueError('native cloud expired during reconstruction')
            if self.source_fault.is_set():return
            self.clouds.publish(msg);self.count+=1;self.error=None
            raw_points,raw_bytes=meta['width']*meta['height'],len(blob)
            self.last_cloud_metrics={'mapper_topic':self.mapper_topic,'mapper_points':None,
                'mapper_raw_bytes':None,'mapper_subset_published':False,'crop_ms':None,
                'mapper_reference_points':0,'raw_points':raw_points,'raw_bytes':raw_bytes}
            mapper_msg=msg;crop_ms=0.;reference_points=0
            if self.mapper_clouds is not None:
                crop_started=time.monotonic()
                keep=mapper_input_mask(decoded[2],msg.header.frame_id,self.cfg)
                offsets=decoded[3]
                if offsets is not None and len(offsets):
                    reference_index=int(offsets.argmax())
                    if not keep[reference_index]:
                        # Keep the real scan endpoint without changing any time.
                        # Pipeline applies the same physical filters afterwards.
                        keep[reference_index]=True;reference_points=1
                mapper_msg=subset_cloud(msg,keep)
                crop_ms=(time.monotonic()-crop_started)*1000
                self.mapper_clouds.publish(mapper_msg)
            mapper_metrics={'mapper_topic':self.mapper_topic,
                'mapper_points':mapper_msg.width*mapper_msg.height,
                'mapper_raw_bytes':len(mapper_msg.data),'mapper_subset_published':True,
                'mapper_reference_points':reference_points,'crop_ms':crop_ms}
            self.last_cloud_metrics={**mapper_metrics,'raw_points':raw_points,'raw_bytes':raw_bytes}
            receipt=String();receipt.data=json.dumps({'protocol':PROTOCOL,'source':SOURCE,'board':'CA9B_GOS',
                'session':self.session,'source_session':meta['native_session'],'sequence':meta['native_sequence'],
                'driver_pid':meta['driver_pid'],'writer_guid':meta['writer_guid'],
                'stamp':meta['sec']+meta['nanosec']*1e-9,'end_stamp':end,
                'received_at':self.get_clock().now().nanoseconds*1e-9,'frame':meta['frame'],
                'points':raw_points,'raw_bytes':raw_bytes,
                'payload_validated':True,'published_local':True,**mapper_metrics})
            self.receipts.publish(receipt)
        except (ValueError,KeyError,TypeError) as exc:self.error=str(exc)

    def imu(self,msg):
        if self.source_fault.is_set():return
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        age=self.get_clock().now().nanoseconds*1e-9-stamp;r=self.cfg['runtime']
        if msg.header.frame_id not in ('','imu_link') or not -r['future_tolerance_s']<=age<=r['input_timeout_s']:
            self.error='native IMU frame/time rejected';return
        msg.header.frame_id='imu_link';self.imus.publish(msg);self.imu_count+=1

    def report(self):
        msg=String();msg.data=json.dumps({'board':'CA9B_GOS','source':SOURCE,'protocol':PROTOCOL,
            'session':self.session,'clouds':self.count,'imus':self.imu_count,
            'factory_identity':self.identity.identity,'last_error':self.error,
            'source_fault_latched':self.source_fault.is_set(),'commands':0,**self.last_cloud_metrics})
        self.status.publish(msg)

    def destroy_node(self):
        self.stop.set();sock=self.sock
        if sock is not None:
            try:sock.shutdown(socket.SHUT_RDWR)
            except OSError:pass
        self.worker.join(timeout=2.)
        return super().destroy_node()


def main():
    rclpy.init();node=None;executor=None
    try:
        node=NativeInputs()
        executor=MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:pass
    finally:
        if executor is not None:executor.shutdown()
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
